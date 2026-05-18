#include <iostream>
#include <windows.h>
#include <immintrin.h>
#include <chrono>
#include <string>

#pragma comment(lib, "Ws2_32.lib")

struct alignas(64) L2Snapshot {
    int32_t prices[8];   // 8 levels of bid prices (32 bytes)
    int32_t volumes[8];  // 8 levels of bid volumes (32 bytes)
};

// Vectorized update: Adds incoming delta delta_v to volumes matching targeted target_p
void update_book_simd(L2Snapshot* book, int32_t target_p, int32_t delta_v) {
    __m256i target_price_vec = _mm256_set1_epi32(target_p);
    __m256i delta_vol_vec    = _mm256_set1_epi32(delta_v);

    __m256i book_prices      = _mm256_load_si256((const __m256i*)book->prices);
    __m256i book_volumes     = _mm256_load_si256((const __m256i*)book->volumes);

    __m256i mask = _mm256_cmpeq_epi32(book_prices, target_price_vec);
    __m256i masked_delta = _mm256_and_si256(mask, delta_vol_vec);

    __m256i updated_volumes = _mm256_add_epi32(book_volumes, masked_delta);
    _mm256_store_si256((__m256i*)book->volumes, updated_volumes);
}

extern "C" {
    // Computes density field snapshots over historical ticks.
    // snapshots_out must be pre-allocated to size: max_snapshots * n_bins
    __declspec(dllexport) int process_empirical_data(
        const char* filename, 
        float* snapshots_out, 
        int max_snapshots, 
        int n_bins, 
        int bin_width, 
        int32_t mid_price, 
        int ticks_per_snapshot
    ) {
        HANDLE hFile = CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            std::cerr << "Failed to open file: " << filename << "\n";
            return -1;
        }

        LARGE_INTEGER fileSize;
        GetFileSizeEx(hFile, &fileSize);

        HANDLE hMap = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (hMap == NULL) {
            CloseHandle(hFile);
            return -1;
        }

        char* mmapped_data = (char*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (mmapped_data == NULL) {
            CloseHandle(hMap);
            CloseHandle(hFile);
            return -1;
        }

        int num_messages = fileSize.QuadPart / 8;
        
        // Active density grid
        float* current_density = new float[n_bins](); // initialized to 0

        int snapshot_count = 0;
        int ticks_since_snapshot = 0;

        for (int i = 0; i < num_messages; ++i) {
            int32_t target_p = *(int32_t*)(mmapped_data + i * 8);
            int32_t delta_v  = *(int32_t*)(mmapped_data + i * 8 + 4);
            
            // Per tick update: binning
            int bin = (int)((target_p - mid_price) / bin_width) + n_bins / 2;
            if (bin >= 0 && bin < n_bins) {
                current_density[bin] += (float)delta_v;
            }
            
            ticks_since_snapshot++;
            
            if (ticks_since_snapshot >= ticks_per_snapshot) {
                if (snapshot_count < max_snapshots) {
                    // Save snapshot
                    for (int b = 0; b < n_bins; ++b) {
                        snapshots_out[snapshot_count * n_bins + b] = current_density[b];
                    }
                    snapshot_count++;
                }
                ticks_since_snapshot = 0;
            }
        }
        
        // Save remaining if partially filled window
        if (ticks_since_snapshot > 0 && snapshot_count < max_snapshots) {
            for (int b = 0; b < n_bins; ++b) {
                snapshots_out[snapshot_count * n_bins + b] = current_density[b];
            }
            snapshot_count++;
        }

        delete[] current_density;
        UnmapViewOfFile(mmapped_data);
        CloseHandle(hMap);
        CloseHandle(hFile);
        
        return snapshot_count;
    }
}

void run_replay_mode(L2Snapshot* book) {
    const char* filename = "historical_itch_feed.bin";
    
    int max_snapshots = 1000;
    int n_bins = 100;
    float* snapshots = new float[max_snapshots * n_bins];
    
    auto start = std::chrono::high_resolution_clock::now();
    int snapshots_generated = process_empirical_data(filename, snapshots, max_snapshots, n_bins, 1000, 6500000, 20);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    
    if (snapshots_generated > 0) {
        std::cout << "[Replay Mode] C++ Density Binning took " << duration.count() << " seconds.\n";
        std::cout << "Generated " << snapshots_generated << " snapshots.\n";
    }
    
    delete[] snapshots;
}

void run_live_mode(L2Snapshot* book) {
    std::cout << "[Live Mode] Initializing UDP multicast socket for live ingestion...\n";
    
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed.\n";
        return;
    }

    SOCKET recvSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (recvSocket == INVALID_SOCKET) {
        std::cerr << "Socket creation failed.\n";
        WSACleanup();
        return;
    }

    sockaddr_in recvAddr;
    recvAddr.sin_family = AF_INET;
    recvAddr.sin_port = htons(12345);
    recvAddr.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(recvSocket, (SOCKADDR*)&recvAddr, sizeof(recvAddr)) == SOCKET_ERROR) {
        std::cerr << "Bind failed.\n";
        closesocket(recvSocket);
        WSACleanup();
        return;
    }

    std::cout << "Listening for live ticks on port 12345...\n";
    
    char buffer[1024];
    while (true) {
        int bytesReceived = recvfrom(recvSocket, buffer, sizeof(buffer), 0, NULL, NULL);
        if (bytesReceived >= 8) {
            for (int i = 0; i < bytesReceived; i += 8) {
                int32_t target_p = *(int32_t*)(buffer + i);
                int32_t delta_v  = *(int32_t*)(buffer + i + 4);
                update_book_simd(book, target_p, delta_v);
            }
        }
    }

    closesocket(recvSocket);
    WSACleanup();
}

int main(int argc, char* argv[]) {
    bool live_mode = false;
    if (argc > 1 && std::string(argv[1]) == "--live") {
        live_mode = true;
    }
    
    // Initialize book
    L2Snapshot book;
    for (int i = 0; i < 8; ++i) {
        book.prices[i] = 100 + i;
        book.volumes[i] = 0;
    }

    if (live_mode) {
        run_live_mode(&book);
    } else {
        run_replay_mode(&book);
    }

    return 0;
}
