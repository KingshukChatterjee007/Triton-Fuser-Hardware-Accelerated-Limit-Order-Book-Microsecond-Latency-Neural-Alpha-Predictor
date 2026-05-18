#include <iostream>
#include <chrono>
#include <string>
#include <immintrin.h>

#ifdef _WIN32
#include <windows.h>
#pragma comment(lib, "Ws2_32.lib")
#define DLLEXPORT __declspec(dllexport)
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
typedef int SOCKET;
#define DLLEXPORT
#endif

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
    DLLEXPORT int process_empirical_data(
        const char* filename, 
        float* snapshots_out, 
        int max_snapshots, 
        int n_bins, 
        int bin_width, 
        int32_t mid_price, 
        int ticks_per_snapshot
    ) {
        char* mmapped_data = nullptr;
        size_t fileSizeResult = 0;
        
        #ifdef _WIN32
        HANDLE hFile = CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (hFile == INVALID_HANDLE_VALUE) {
            std::cerr << "Failed to open file: " << filename << "\n";
            return -1;
        }
        LARGE_INTEGER fileSize;
        GetFileSizeEx(hFile, &fileSize);
        fileSizeResult = fileSize.QuadPart;

        HANDLE hMap = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
        if (hMap == NULL) {
            CloseHandle(hFile);
            return -1;
        }
        mmapped_data = (char*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (mmapped_data == NULL) {
            CloseHandle(hMap);
            CloseHandle(hFile);
            return -1;
        }
        #else
        int fd = open(filename, O_RDONLY);
        if (fd < 0) {
            std::cerr << "Failed to open file: " << filename << "\n";
            return -1;
        }
        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            close(fd);
            return -1;
        }
        fileSizeResult = sb.st_size;
        mmapped_data = (char*)mmap(NULL, fileSizeResult, PROT_READ, MAP_SHARED, fd, 0);
        if (mmapped_data == MAP_FAILED) {
            close(fd);
            return -1;
        }
        #endif

        int num_messages = fileSizeResult / 8;
        
        // Active density grid
        float* current_density = new float[n_bins](); // initialized to 0

        int snapshot_count = 0;
        int ticks_since_snapshot = 0;

        int i = 0;
        // ── GENUINE AVX2 SIMD VECTORIZED PARALLEL BINNING ──
        for (; i <= num_messages - 8; i += 8) {
            // Load 8 interleaved elements: Price (4 bytes) and Volume (4 bytes)
            // Indices of prices in 8-byte steps: 0, 8, 16, 24, 32, 40, 48, 56
            // Indices of volumes in 8-byte steps: 4, 12, 20, 28, 36, 44, 52, 60
            __m256i price_offsets = _mm256_setr_epi32(0*8, 1*8, 2*8, 3*8, 4*8, 5*8, 6*8, 7*8);
            __m256i vol_offsets   = _mm256_setr_epi32(0*8 + 4, 1*8 + 4, 2*8 + 4, 3*8 + 4, 4*8 + 4, 5*8 + 4, 6*8 + 4, 7*8 + 4);
            
            // Gather 8 prices and volumes in parallel using Vector Register Gathers
            __m256i prices_vec = _mm256_i32gather_epi32((const int*)(mmapped_data + i * 8), price_offsets, 1);
            __m256i vols_vec   = _mm256_i32gather_epi32((const int*)(mmapped_data + i * 8), vol_offsets, 1);
            
            __m256i mid_price_vec = _mm256_set1_epi32(mid_price);
            __m256i diff = _mm256_sub_epi32(prices_vec, mid_price_vec);
            
            // Convert to floats for vector division (no native int32 division in AVX2)
            __m256 float_diff = _mm256_cvtepi32_ps(diff);
            __m256 float_width = _mm256_set1_ps((float)bin_width);
            __m256 float_bin = _mm256_div_ps(float_diff, float_width);
            __m256i bin_indices = _mm256_cvtps_epi32(float_bin);
            
            __m256i offset_vec = _mm256_set1_epi32(n_bins / 2);
            bin_indices = _mm256_add_epi32(bin_indices, offset_vec);
            
            alignas(32) int32_t indices[8];
            alignas(32) int32_t volumes[8];
            _mm256_store_si256((__m256i*)indices, bin_indices);
            _mm256_store_si256((__m256i*)volumes, vols_vec);
            
            for (int k = 0; k < 8; ++k) {
                int bin = indices[k];
                if (bin >= 0 && bin < n_bins) {
                    current_density[bin] += (float)volumes[k];
                }
                
                ticks_since_snapshot++;
                if (ticks_since_snapshot >= ticks_per_snapshot) {
                    if (snapshot_count < max_snapshots) {
                        for (int b = 0; b < n_bins; ++b) {
                            snapshots_out[snapshot_count * n_bins + b] = current_density[b];
                        }
                        snapshot_count++;
                    }
                    ticks_since_snapshot = 0;
                }
            }
        }
        
        // ── SCALAR FALLBACK FOR REMAINING ELEMENTS ──
        for (; i < num_messages; ++i) {
            int32_t target_p = *(int32_t*)(mmapped_data + i * 8);
            int32_t delta_v  = *(int32_t*)(mmapped_data + i * 8 + 4);
            
            int bin = (int)((target_p - mid_price) / bin_width) + n_bins / 2;
            if (bin >= 0 && bin < n_bins) {
                current_density[bin] += (float)delta_v;
            }
            
            ticks_since_snapshot++;
            if (ticks_since_snapshot >= ticks_per_snapshot) {
                if (snapshot_count < max_snapshots) {
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

        #ifdef _WIN32
        UnmapViewOfFile(mmapped_data);
        CloseHandle(hMap);
        CloseHandle(hFile);
        #else
        munmap(mmapped_data, fileSizeResult);
        close(fd);
        #endif
        
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
    
    #ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        std::cerr << "WSAStartup failed.\n";
        return;
    }
    #endif

    SOCKET recvSocket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (recvSocket == INVALID_SOCKET) {
        std::cerr << "Socket creation failed.\n";
        #ifdef _WIN32
        WSACleanup();
        #endif
        return;
    }

    sockaddr_in recvAddr;
    recvAddr.sin_family = AF_INET;
    recvAddr.sin_port = htons(12345);
    recvAddr.sin_addr.s_addr = htonl(INADDR_ANY);

    #ifdef _WIN32
    if (bind(recvSocket, (SOCKADDR*)&recvAddr, sizeof(recvAddr)) == SOCKET_ERROR) {
    #else
    if (bind(recvSocket, (struct sockaddr*)&recvAddr, sizeof(recvAddr)) == SOCKET_ERROR) {
    #endif
        std::cerr << "Bind failed.\n";
        #ifdef _WIN32
        closesocket(recvSocket);
        WSACleanup();
        #else
        close(recvSocket);
        #endif
        return;
    }

    std::cout << "Listening for live ticks on port 12345...\n";
    
    char buffer[1024];
    while (true) {
        #ifdef _WIN32
        int bytesReceived = recvfrom(recvSocket, buffer, sizeof(buffer), 0, NULL, NULL);
        #else
        int bytesReceived = recvfrom(recvSocket, buffer, sizeof(buffer), 0, NULL, NULL);
        #endif
        if (bytesReceived >= 8) {
            for (int i = 0; i < bytesReceived; i += 8) {
                int32_t target_p = *(int32_t*)(buffer + i);
                int32_t delta_v  = *(int32_t*)(buffer + i + 4);
                update_book_simd(book, target_p, delta_v);
            }
        }
    }

    #ifdef _WIN32
    closesocket(recvSocket);
    WSACleanup();
    #else
    close(recvSocket);
    #endif
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
