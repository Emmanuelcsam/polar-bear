# Complete Guide: Raspberry Pi Pico 2 Image Capture System for Dimension Scope EC200KC V2

## System Overview

This guide provides meticulous step-by-step instructions for building a system that captures images from a Dimension Scope Easy Check V2 Model EC200KC V2 and transfers them to a laptop using two Raspberry Pi Pico 2 boards.

## Hardware Requirements

### Components Needed
- 2× Raspberry Pi Pico 2 boards (RP2350, 520KB RAM)
- 1× MicroSD card breakout module (3.3V compatible)
- Jumper wires
- USB cables (micro-USB)
- Optional: Logic level analyzer for debugging

### Key Specifications
- **Dimension Scope**: 3.3V SD card interface, saves images in JPEG/BMP format
- **Pico 2**: Dual-core ARM Cortex-M33 @ 150MHz, 520KB SRAM, 4MB Flash
- **Communication**: SPI between Picos (up to 20MHz), USB CDC to laptop

## Part 1: Connecting Pico 2 A to SD Card Slot

### Step 1: Hardware Connections

**SD Card Module to Pico 2 A Wiring:**
```
SD Card Module → Raspberry Pi Pico 2 A
VCC  → 3V3 (Pin 36)     [3.3V power]
GND  → GND (Pin 38)     [Ground]
MISO → GPIO 16 (Pin 21) [SPI0 RX]
MOSI → GPIO 19 (Pin 25) [SPI0 TX]
SCK  → GPIO 18 (Pin 24) [SPI0 SCK]
CS   → GPIO 17 (Pin 22) [SPI0 CSn]
```

**Important:** Use a 3.3V SD card module. Do NOT use 5V Arduino modules as they can damage the Pico.

### Step 2: Install Required Software

**Windows Setup:**
1. Download and install the Pico SDK installer from https://github.com/raspberrypi/pico-setup-windows/releases
2. Install Visual Studio Code
3. Install the Pico extension in VS Code

**Linux/Mac Setup:**
```bash
# Install dependencies
sudo apt update
sudo apt install cmake gcc-arm-none-eabi libnewlib-arm-none-eabi build-essential

# Clone Pico SDK
git clone https://github.com/raspberrypi/pico-sdk.git
cd pico-sdk
git submodule update --init
export PICO_SDK_PATH=$PWD
```

### Step 3: Create Project Structure

Create a new directory for your project:
```bash
mkdir pico-image-transfer
cd pico-image-transfer
mkdir pico_a pico_b computer_side
```

## Part 2: Pico A - SD Card Reading and Image Detection

### Step 1: Download Required Libraries

```bash
cd pico_a
# Download sdcard.py for MicroPython implementation
wget https://raw.githubusercontent.com/micropython/micropython/master/drivers/sdcard/sdcard.py
```

### Step 2: Create Main Program for Pico A

Create `CMakeLists.txt`:
```cmake
cmake_minimum_required(VERSION 3.13)

include(pico_sdk_import.cmake)

project(pico_a_sd_reader C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

add_executable(pico_a_sd_reader
    main.c
    sd_card.c
    image_handler.c
    spi_transfer.c
)

pico_enable_stdio_usb(pico_a_sd_reader 1)
pico_enable_stdio_uart(pico_a_sd_reader 0)

target_link_libraries(pico_a_sd_reader
    pico_stdlib
    hardware_spi
    hardware_dma
    hardware_timer
)

pico_add_extra_outputs(pico_a_sd_reader)
```

### Step 3: SD Card Interface Code

Create `sd_card.c`:
```c
#include "pico/stdlib.h"
#include "hardware/spi.h"
#include "hardware/gpio.h"
#include <string.h>
#include <stdio.h>

// SD Card Commands
#define CMD0    0x40    // GO_IDLE_STATE
#define CMD8    0x48    // SEND_IF_COND
#define CMD55   0x77    // APP_CMD
#define ACMD41  0x69    // SD_SEND_OP_COND
#define CMD58   0x7A    // READ_OCR
#define CMD17   0x51    // READ_SINGLE_BLOCK
#define CMD24   0x58    // WRITE_SINGLE_BLOCK

// Pin definitions
#define SPI_PORT spi0
#define PIN_MISO 16
#define PIN_CS   17
#define PIN_SCK  18
#define PIN_MOSI 19

static uint8_t sd_buffer[512];
static bool sd_initialized = false;

// Initialize SPI for SD card
void sd_spi_init(void) {
    spi_init(SPI_PORT, 400000); // Start at 400kHz for initialization
    
    gpio_set_function(PIN_MISO, GPIO_FUNC_SPI);
    gpio_set_function(PIN_SCK, GPIO_FUNC_SPI);
    gpio_set_function(PIN_MOSI, GPIO_FUNC_SPI);
    
    // Configure CS pin
    gpio_init(PIN_CS);
    gpio_set_dir(PIN_CS, GPIO_OUT);
    gpio_put(PIN_CS, 1); // CS high (inactive)
    
    // Add pull-up on MISO
    gpio_pull_up(PIN_MISO);
}

// Send command to SD card
uint8_t sd_send_command(uint8_t cmd, uint32_t arg) {
    uint8_t response;
    uint8_t crc = 0x01; // Default CRC
    
    if (cmd == CMD0) crc = 0x95;
    if (cmd == CMD8) crc = 0x87;
    
    // Select card
    gpio_put(PIN_CS, 0);
    
    // Send command
    uint8_t tx_buffer[6];
    tx_buffer[0] = cmd;
    tx_buffer[1] = (arg >> 24) & 0xFF;
    tx_buffer[2] = (arg >> 16) & 0xFF;
    tx_buffer[3] = (arg >> 8) & 0xFF;
    tx_buffer[4] = arg & 0xFF;
    tx_buffer[5] = crc;
    
    spi_write_blocking(SPI_PORT, tx_buffer, 6);
    
    // Wait for response
    for (int i = 0; i < 10; i++) {
        spi_read_blocking(SPI_PORT, 0xFF, &response, 1);
        if ((response & 0x80) == 0) break;
    }
    
    return response;
}

// Initialize SD card
bool sd_init(void) {
    sd_spi_init();
    sleep_ms(10);
    
    // Send 80 clock pulses
    uint8_t dummy[10];
    memset(dummy, 0xFF, 10);
    gpio_put(PIN_CS, 1);
    spi_write_blocking(SPI_PORT, dummy, 10);
    
    // CMD0: GO_IDLE_STATE
    uint8_t response = sd_send_command(CMD0, 0);
    gpio_put(PIN_CS, 1);
    
    if (response != 0x01) {
        printf("SD Card: CMD0 failed (response: 0x%02X)\n", response);
        return false;
    }
    
    // CMD8: SEND_IF_COND
    response = sd_send_command(CMD8, 0x1AA);
    uint8_t ocr[4];
    spi_read_blocking(SPI_PORT, 0xFF, ocr, 4);
    gpio_put(PIN_CS, 1);
    
    // Initialize card with ACMD41
    for (int i = 0; i < 100; i++) {
        sd_send_command(CMD55, 0);
        gpio_put(PIN_CS, 1);
        
        response = sd_send_command(ACMD41, 0x40000000);
        gpio_put(PIN_CS, 1);
        
        if (response == 0x00) break;
        sleep_ms(10);
    }
    
    if (response != 0x00) {
        printf("SD Card: Initialization failed\n");
        return false;
    }
    
    // Increase SPI speed
    spi_set_baudrate(SPI_PORT, 10000000); // 10MHz
    
    sd_initialized = true;
    printf("SD Card: Initialized successfully\n");
    return true;
}

// Read a 512-byte block
bool sd_read_block(uint32_t block_addr, uint8_t *buffer) {
    if (!sd_initialized) return false;
    
    uint8_t response = sd_send_command(CMD17, block_addr);
    
    if (response != 0x00) {
        gpio_put(PIN_CS, 1);
        return false;
    }
    
    // Wait for data token
    uint8_t token;
    for (int i = 0; i < 1000; i++) {
        spi_read_blocking(SPI_PORT, 0xFF, &token, 1);
        if (token == 0xFE) break;
    }
    
    if (token != 0xFE) {
        gpio_put(PIN_CS, 1);
        return false;
    }
    
    // Read data
    spi_read_blocking(SPI_PORT, 0xFF, buffer, 512);
    
    // Read CRC (ignored)
    uint8_t crc[2];
    spi_read_blocking(SPI_PORT, 0xFF, crc, 2);
    
    gpio_put(PIN_CS, 1);
    return true;
}
```

### Step 4: Image Detection and Monitoring

Create `image_handler.c`:
```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "pico/stdlib.h"
#include "image_handler.h"

#define FAT32_BOOT_SECTOR 0
#define BYTES_PER_SECTOR 512
#define MAX_FILES 100

typedef struct {
    char filename[256];
    uint32_t size;
    uint32_t first_cluster;
    uint32_t modify_time;
} FileInfo;

typedef struct {
    uint32_t sectors_per_cluster;
    uint32_t reserved_sectors;
    uint32_t fat_start_sector;
    uint32_t data_start_sector;
    uint32_t root_dir_cluster;
} FAT32_Info;

static FileInfo known_files[MAX_FILES];
static int known_file_count = 0;
static FAT32_Info fat_info;

// Parse FAT32 boot sector
bool parse_boot_sector(uint8_t *buffer) {
    // Check FAT32 signature
    if (buffer[510] != 0x55 || buffer[511] != 0xAA) {
        return false;
    }
    
    // Extract FAT32 parameters
    fat_info.sectors_per_cluster = buffer[13];
    fat_info.reserved_sectors = *(uint16_t*)&buffer[14];
    uint8_t num_fats = buffer[16];
    uint32_t sectors_per_fat = *(uint32_t*)&buffer[36];
    
    fat_info.fat_start_sector = fat_info.reserved_sectors;
    fat_info.data_start_sector = fat_info.fat_start_sector + 
                                 (num_fats * sectors_per_fat);
    fat_info.root_dir_cluster = *(uint32_t*)&buffer[44];
    
    return true;
}

// Monitor for new images
bool check_for_new_images(void) {
    static uint32_t last_check_time = 0;
    uint32_t current_time = time_us_32() / 1000000; // Convert to seconds
    
    // Check every 2 seconds
    if (current_time - last_check_time < 2) {
        return false;
    }
    
    last_check_time = current_time;
    
    // Read root directory
    uint32_t cluster = fat_info.root_dir_cluster;
    uint32_t sector = ((cluster - 2) * fat_info.sectors_per_cluster) + 
                      fat_info.data_start_sector;
    
    uint8_t buffer[512];
    if (!sd_read_block(sector * 512, buffer)) {
        return false;
    }
    
    // Parse directory entries
    bool new_file_found = false;
    for (int i = 0; i < 512; i += 32) {
        if (buffer[i] == 0x00) break; // End of directory
        if (buffer[i] == 0xE5) continue; // Deleted entry
        
        // Check if it's a valid file
        if ((buffer[i + 11] & 0x10) == 0) { // Not a directory
            char filename[13];
            memcpy(filename, &buffer[i], 11);
            filename[11] = '\0';
            
            // Check if it's an image file (JPG or BMP)
            if (strstr(filename, "JPG") || strstr(filename, "BMP")) {
                // Check if this is a new file
                bool is_new = true;
                for (int j = 0; j < known_file_count; j++) {
                    if (strcmp(known_files[j].filename, filename) == 0) {
                        is_new = false;
                        break;
                    }
                }
                
                if (is_new && known_file_count < MAX_FILES) {
                    strcpy(known_files[known_file_count].filename, filename);
                    known_files[known_file_count].size = *(uint32_t*)&buffer[i + 28];
                    known_files[known_file_count].first_cluster = 
                        (*(uint16_t*)&buffer[i + 26]) | 
                        ((uint32_t)(*(uint16_t*)&buffer[i + 20]) << 16);
                    known_file_count++;
                    new_file_found = true;
                    printf("New image detected: %s (%lu bytes)\n", 
                           filename, known_files[known_file_count-1].size);
                }
            }
        }
    }
    
    return new_file_found;
}

// Get latest image info
bool get_latest_image(FileInfo **file_info) {
    if (known_file_count == 0) return false;
    
    *file_info = &known_files[known_file_count - 1];
    return true;
}

// Read image data in chunks
bool read_image_chunk(FileInfo *file, uint32_t offset, uint8_t *buffer, 
                     uint32_t *bytes_read, uint32_t max_size) {
    uint32_t cluster = file->first_cluster;
    uint32_t bytes_per_cluster = fat_info.sectors_per_cluster * 512;
    
    // Calculate which cluster contains the offset
    uint32_t cluster_offset = offset / bytes_per_cluster;
    uint32_t byte_offset = offset % bytes_per_cluster;
    
    // Navigate to the correct cluster
    for (uint32_t i = 0; i < cluster_offset; i++) {
        // Read FAT to get next cluster
        uint32_t fat_sector = fat_info.fat_start_sector + (cluster * 4) / 512;
        uint32_t fat_offset = (cluster * 4) % 512;
        
        uint8_t fat_buffer[512];
        if (!sd_read_block(fat_sector * 512, fat_buffer)) {
            return false;
        }
        
        cluster = *(uint32_t*)&fat_buffer[fat_offset] & 0x0FFFFFFF;
        if (cluster >= 0x0FFFFFF8) { // End of chain
            return false;
        }
    }
    
    // Read data from cluster
    uint32_t sector = ((cluster - 2) * fat_info.sectors_per_cluster) + 
                      fat_info.data_start_sector;
    uint32_t sector_offset = byte_offset / 512;
    uint32_t byte_in_sector = byte_offset % 512;
    
    uint8_t temp_buffer[512];
    uint32_t total_read = 0;
    
    while (total_read < max_size && offset + total_read < file->size) {
        if (!sd_read_block((sector + sector_offset) * 512, temp_buffer)) {
            return false;
        }
        
        uint32_t to_copy = 512 - byte_in_sector;
        if (to_copy > max_size - total_read) {
            to_copy = max_size - total_read;
        }
        if (offset + total_read + to_copy > file->size) {
            to_copy = file->size - offset - total_read;
        }
        
        memcpy(buffer + total_read, temp_buffer + byte_in_sector, to_copy);
        total_read += to_copy;
        
        byte_in_sector = 0;
        sector_offset++;
        
        // Check if we need to move to next cluster
        if (sector_offset >= fat_info.sectors_per_cluster) {
            // Get next cluster from FAT
            uint32_t fat_sector = fat_info.fat_start_sector + (cluster * 4) / 512;
            uint32_t fat_offset = (cluster * 4) % 512;
            
            if (!sd_read_block(fat_sector * 512, temp_buffer)) {
                return false;
            }
            
            cluster = *(uint32_t*)&temp_buffer[fat_offset] & 0x0FFFFFFF;
            if (cluster >= 0x0FFFFFF8) { // End of chain
                break;
            }
            
            sector = ((cluster - 2) * fat_info.sectors_per_cluster) + 
                     fat_info.data_start_sector;
            sector_offset = 0;
        }
    }
    
    *bytes_read = total_read;
    return true;
}
```

## Part 3: SPI Communication Between Pico Boards

### Step 1: Hardware Connections Between Picos

```
Pico 2 A (Master) → Pico 2 B (Slave)
GPIO 18 (SCK)     → GPIO 18 (SCK)
GPIO 19 (MOSI)    → GPIO 19 (MOSI)
GPIO 16 (MISO)    → GPIO 16 (MISO)
GPIO 17 (CS)      → GPIO 17 (CS)
GND               → GND
```

### Step 2: SPI Transfer Protocol Implementation

Create `spi_transfer.c` for Pico A:
```c
#include "pico/stdlib.h"
#include "hardware/spi.h"
#include "hardware/dma.h"
#include "hardware/irq.h"

#define SPI_TRANSFER_PORT spi0
#define SPI_BAUDRATE 10000000 // 10MHz
#define CHUNK_SIZE 2048
#define ACK_TIMEOUT_MS 1000

// Packet types
#define PACKET_HEADER 0x01
#define PACKET_DATA   0x02
#define PACKET_END    0x03
#define PACKET_ACK    0x06
#define PACKET_NAK    0x15
#define PACKET_RESEND 0x12

typedef struct {
    uint8_t type;
    uint16_t sequence;
    uint16_t length;
    uint32_t total_size;
    uint32_t crc32;
} packet_header_t;

static uint32_t crc32_table[256];
static bool crc_table_initialized = false;

// Initialize CRC32 table
void init_crc32_table(void) {
    if (crc_table_initialized) return;
    
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            if (crc & 1) {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
        crc32_table[i] = crc;
    }
    crc_table_initialized = true;
}

// Calculate CRC32
uint32_t calculate_crc32(const uint8_t *data, size_t length) {
    init_crc32_table();
    uint32_t crc = 0xFFFFFFFF;
    
    for (size_t i = 0; i < length; i++) {
        crc = crc32_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    
    return crc ^ 0xFFFFFFFF;
}

// Initialize SPI as master
void spi_master_init(void) {
    spi_init(SPI_TRANSFER_PORT, SPI_BAUDRATE);
    
    gpio_set_function(16, GPIO_FUNC_SPI); // MISO
    gpio_set_function(18, GPIO_FUNC_SPI); // SCK
    gpio_set_function(19, GPIO_FUNC_SPI); // MOSI
    
    // Configure CS pin
    gpio_init(17);
    gpio_set_dir(17, GPIO_OUT);
    gpio_put(17, 1); // CS high (inactive)
}

// Send packet with retry
bool send_packet_with_retry(packet_header_t *header, uint8_t *data, int retries) {
    uint8_t ack_buffer[16];
    
    for (int attempt = 0; attempt < retries; attempt++) {
        // Select slave
        gpio_put(17, 0);
        sleep_us(10);
        
        // Send header
        spi_write_blocking(SPI_TRANSFER_PORT, (uint8_t*)header, sizeof(packet_header_t));
        
        // Send data if present
        if (data && header->length > 0) {
            spi_write_blocking(SPI_TRANSFER_PORT, data, header->length);
        }
        
        // Deselect slave
        gpio_put(17, 1);
        sleep_ms(1);
        
        // Wait for acknowledgment
        gpio_put(17, 0);
        spi_read_blocking(SPI_TRANSFER_PORT, 0xFF, ack_buffer, 1);
        gpio_put(17, 1);
        
        if (ack_buffer[0] == PACKET_ACK) {
            return true;
        } else if (ack_buffer[0] == PACKET_NAK) {
            printf("NAK received, retrying %d/%d\n", attempt + 1, retries);
            sleep_ms(10 * (attempt + 1)); // Exponential backoff
        }
    }
    
    return false;
}

// Transfer complete image
bool transfer_image(uint8_t *image_data, uint32_t image_size) {
    printf("Starting image transfer: %lu bytes\n", image_size);
    
    // Send header packet
    packet_header_t header;
    header.type = PACKET_HEADER;
    header.sequence = 0;
    header.length = 0;
    header.total_size = image_size;
    header.crc32 = calculate_crc32(image_data, image_size);
    
    if (!send_packet_with_retry(&header, NULL, 3)) {
        printf("Failed to send header\n");
        return false;
    }
    
    // Send data in chunks
    uint32_t offset = 0;
    uint16_t sequence = 1;
    
    while (offset < image_size) {
        uint32_t chunk_size = (image_size - offset > CHUNK_SIZE) ? 
                              CHUNK_SIZE : (image_size - offset);
        
        header.type = PACKET_DATA;
        header.sequence = sequence++;
        header.length = chunk_size;
        header.total_size = image_size;
        header.crc32 = calculate_crc32(&image_data[offset], chunk_size);
        
        if (!send_packet_with_retry(&header, &image_data[offset], 3)) {
            printf("Failed to send chunk at offset %lu\n", offset);
            return false;
        }
        
        offset += chunk_size;
        
        // Progress update
        uint32_t progress = (offset * 100) / image_size;
        printf("Progress: %lu%% (%lu/%lu bytes)\n", progress, offset, image_size);
    }
    
    // Send end packet
    header.type = PACKET_END;
    header.sequence = sequence;
    header.length = 0;
    header.total_size = image_size;
    header.crc32 = 0;
    
    if (!send_packet_with_retry(&header, NULL, 3)) {
        printf("Failed to send end packet\n");
        return false;
    }
    
    printf("Image transfer completed successfully!\n");
    return true;
}

// Main function for Pico A
int main() {
    stdio_init_all();
    sleep_ms(2000); // Wait for USB enumeration
    
    printf("Pico A: SD Card Reader and Image Sender\n");
    
    // Initialize SD card
    if (!sd_init()) {
        printf("Failed to initialize SD card\n");
        while (1) sleep_ms(1000);
    }
    
    // Parse FAT32 filesystem
    uint8_t boot_sector[512];
    if (!sd_read_block(0, boot_sector)) {
        printf("Failed to read boot sector\n");
        while (1) sleep_ms(1000);
    }
    
    if (!parse_boot_sector(boot_sector)) {
        printf("Failed to parse boot sector\n");
        while (1) sleep_ms(1000);
    }
    
    // Initialize SPI for Pico-to-Pico communication
    spi_master_init();
    
    // Main loop - monitor for new images
    while (1) {
        if (check_for_new_images()) {
            FileInfo *new_image;
            if (get_latest_image(&new_image)) {
                printf("Processing new image: %s (%lu bytes)\n", 
                       new_image->filename, new_image->size);
                
                // Allocate buffer for image (or use streaming)
                uint8_t *image_buffer = malloc(new_image->size);
                if (!image_buffer) {
                    printf("Failed to allocate memory for image\n");
                    continue;
                }
                
                // Read complete image
                uint32_t total_read = 0;
                uint32_t offset = 0;
                
                while (offset < new_image->size) {
                    uint32_t bytes_read;
                    uint32_t to_read = (new_image->size - offset > 4096) ? 
                                       4096 : (new_image->size - offset);
                    
                    if (!read_image_chunk(new_image, offset, 
                                         &image_buffer[offset], 
                                         &bytes_read, to_read)) {
                        printf("Failed to read image data\n");
                        break;
                    }
                    
                    offset += bytes_read;
                    total_read += bytes_read;
                }
                
                if (total_read == new_image->size) {
                    // Transfer image to Pico B
                    transfer_image(image_buffer, new_image->size);
                }
                
                free(image_buffer);
            }
        }
        
        sleep_ms(100);
    }
    
    return 0;
}
```

## Part 4: Pico B - Receiving and USB Transfer

### Step 1: Create Pico B Project

Create `CMakeLists.txt` for Pico B:
```cmake
cmake_minimum_required(VERSION 3.13)

include(pico_sdk_import.cmake)

project(pico_b_receiver C CXX ASM)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

pico_sdk_init()

add_executable(pico_b_receiver
    main.c
    spi_receiver.c
    usb_transfer.c
)

pico_enable_stdio_usb(pico_b_receiver 1)
pico_enable_stdio_uart(pico_b_receiver 0)

target_link_libraries(pico_b_receiver
    pico_stdlib
    hardware_spi
    hardware_dma
)

pico_add_extra_outputs(pico_b_receiver)
```

### Step 2: SPI Receiver Implementation

Create `spi_receiver.c` for Pico B:
```c
#include "pico/stdlib.h"
#include "hardware/spi.h"
#include "hardware/gpio.h"
#include <string.h>
#include <stdio.h>

#define SPI_TRANSFER_PORT spi0
#define BUFFER_SIZE 65536 // 64KB buffer
#define PACKET_TIMEOUT_MS 5000

// Reuse packet structures from Pico A
typedef struct {
    uint8_t type;
    uint16_t sequence;
    uint16_t length;
    uint32_t total_size;
    uint32_t crc32;
} packet_header_t;

static uint8_t image_buffer[BUFFER_SIZE];
static uint32_t image_size = 0;
static uint32_t bytes_received = 0;
static bool transfer_complete = false;

// CRC32 implementation (same as Pico A)
static uint32_t crc32_table[256];
static bool crc_table_initialized = false;

void init_crc32_table(void) {
    if (crc_table_initialized) return;
    
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            if (crc & 1) {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
        crc32_table[i] = crc;
    }
    crc_table_initialized = true;
}

uint32_t calculate_crc32(const uint8_t *data, size_t length) {
    init_crc32_table();
    uint32_t crc = 0xFFFFFFFF;
    
    for (size_t i = 0; i < length; i++) {
        crc = crc32_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    
    return crc ^ 0xFFFFFFFF;
}

// Initialize SPI as slave
void spi_slave_init(void) {
    spi_init(SPI_TRANSFER_PORT, 10000000); // 10MHz
    spi_set_slave(SPI_TRANSFER_PORT, true);
    
    gpio_set_function(16, GPIO_FUNC_SPI); // MISO
    gpio_set_function(18, GPIO_FUNC_SPI); // SCK
    gpio_set_function(19, GPIO_FUNC_SPI); // MOSI
    gpio_set_function(17, GPIO_FUNC_SPI); // CS
}

// Receive packet
bool receive_packet(packet_header_t *header, uint8_t *data_buffer) {
    static uint8_t rx_buffer[sizeof(packet_header_t) + 2048];
    uint32_t start_time = time_us_32();
    
    // Wait for CS to go low
    while (gpio_get(17) == 1) {
        if ((time_us_32() - start_time) > (PACKET_TIMEOUT_MS * 1000)) {
            return false;
        }
        sleep_us(10);
    }
    
    // Read header
    spi_read_blocking(SPI_TRANSFER_PORT, 0xFF, (uint8_t*)header, sizeof(packet_header_t));
    
    // Read data if present
    if (header->length > 0 && header->length <= 2048) {
        spi_read_blocking(SPI_TRANSFER_PORT, 0xFF, data_buffer, header->length);
    }
    
    // Wait for CS to go high
    while (gpio_get(17) == 0) {
        sleep_us(10);
    }
    
    // Verify CRC if data present
    if (header->length > 0) {
        uint32_t calc_crc = calculate_crc32(data_buffer, header->length);
        if (calc_crc != header->crc32) {
            return false;
        }
    }
    
    return true;
}

// Send acknowledgment
void send_ack(bool success) {
    uint8_t response = success ? PACKET_ACK : PACKET_NAK;
    
    // Wait for master to request ACK
    while (gpio_get(17) == 1) {
        sleep_us(10);
    }
    
    spi_write_blocking(SPI_TRANSFER_PORT, &response, 1);
    
    // Wait for CS to go high
    while (gpio_get(17) == 0) {
        sleep_us(10);
    }
}

// Process received packets
void process_spi_packets(void) {
    packet_header_t header;
    static uint8_t data_buffer[2048];
    
    while (!transfer_complete) {
        if (receive_packet(&header, data_buffer)) {
            switch (header.type) {
                case PACKET_HEADER:
                    image_size = header.total_size;
                    bytes_received = 0;
                    printf("Starting image reception: %lu bytes\n", image_size);
                    send_ack(true);
                    break;
                    
                case PACKET_DATA:
                    if (bytes_received + header.length <= BUFFER_SIZE) {
                        memcpy(&image_buffer[bytes_received], data_buffer, header.length);
                        bytes_received += header.length;
                        
                        uint32_t progress = (bytes_received * 100) / image_size;
                        printf("Progress: %lu%% (%lu/%lu bytes)\n", 
                               progress, bytes_received, image_size);
                        
                        send_ack(true);
                    } else {
                        printf("Buffer overflow!\n");
                        send_ack(false);
                    }
                    break;
                    
                case PACKET_END:
                    if (bytes_received == image_size) {
                        transfer_complete = true;
                        printf("Image reception complete!\n");
                        send_ack(true);
                    } else {
                        printf("Size mismatch: expected %lu, received %lu\n", 
                               image_size, bytes_received);
                        send_ack(false);
                    }
                    break;
                    
                default:
                    send_ack(false);
                    break;
            }
        }
    }
}
```

### Step 3: USB Transfer to Computer

Create `usb_transfer.c`:
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pico/stdlib.h"

// USB protocol constants
#define USB_START_MARKER 0x55
#define USB_SYNC_BYTE 0x3C
#define USB_PACKET_HEADER 0x01
#define USB_PACKET_DATA 0x02
#define USB_PACKET_FOOTER 0x03
#define USB_ACK_BYTE 0x06
#define USB_NAK_BYTE 0x15
#define USB_MAX_DATA_SIZE 248

typedef struct {
    uint8_t start;
    uint8_t sync;
    uint8_t type;
    uint8_t length;
    uint8_t data[USB_MAX_DATA_SIZE];
    uint8_t checksum;
    uint8_t end;
} usb_packet_t;

// Fletcher-16 checksum
uint16_t fletcher16(uint8_t *data, int len) {
    uint16_t sum1 = 0, sum2 = 0;
    for (int i = 0; i < len; i++) {
        sum1 = (sum1 + data[i]) % 255;
        sum2 = (sum2 + sum1) % 255;
    }
    return (sum2 << 8) | sum1;
}

// Send USB packet
void send_usb_packet(uint8_t type, uint8_t *data, uint8_t length) {
    usb_packet_t packet;
    packet.start = USB_START_MARKER;
    packet.sync = USB_SYNC_BYTE;
    packet.type = type;
    packet.length = length;
    
    if (length > 0 && data != NULL) {
        memcpy(packet.data, data, length);
    }
    
    // Calculate checksum
    uint8_t checksum_data[3 + length];
    checksum_data[0] = packet.sync;
    checksum_data[1] = packet.type;
    checksum_data[2] = packet.length;
    if (length > 0) {
        memcpy(&checksum_data[3], packet.data, length);
    }
    
    packet.checksum = fletcher16(checksum_data, 3 + length) & 0xFF;
    packet.end = USB_START_MARKER;
    
    // Send packet via USB CDC
    fwrite(&packet, 1, 6 + length, stdout);
    fflush(stdout);
}

// Wait for USB acknowledgment
bool wait_for_usb_ack(uint32_t timeout_ms) {
    uint32_t start_time = time_us_32();
    
    while ((time_us_32() - start_time) < (timeout_ms * 1000)) {
        int c = getchar_timeout_us(1000);
        if (c != PICO_ERROR_TIMEOUT) {
            return (c == USB_ACK_BYTE);
        }
    }
    
    return false;
}

// Transfer image to computer via USB
bool transfer_image_to_computer(uint8_t *image_data, uint32_t image_size) {
    printf("\nStarting USB transfer to computer...\n");
    
    // Send header with image size
    uint8_t header_data[4];
    header_data[0] = image_size & 0xFF;
    header_data[1] = (image_size >> 8) & 0xFF;
    header_data[2] = (image_size >> 16) & 0xFF;
    header_data[3] = (image_size >> 24) & 0xFF;
    
    send_usb_packet(USB_PACKET_HEADER, header_data, 4);
    
    if (!wait_for_usb_ack(5000)) {
        printf("Failed to receive header ACK\n");
        return false;
    }
    
    // Send data in chunks
    uint32_t offset = 0;
    while (offset < image_size) {
        uint32_t chunk_size = (image_size - offset > USB_MAX_DATA_SIZE) ?
                              USB_MAX_DATA_SIZE : (image_size - offset);
        
        send_usb_packet(USB_PACKET_DATA, &image_data[offset], chunk_size);
        
        if (!wait_for_usb_ack(5000)) {
            printf("Failed to receive data ACK at offset %lu\n", offset);
            return false;
        }
        
        offset += chunk_size;
        
        // Progress update
        uint32_t progress = (offset * 100) / image_size;
        if (progress % 10 == 0) {
            printf("USB Progress: %lu%%\n", progress);
        }
    }
    
    // Send footer
    send_usb_packet(USB_PACKET_FOOTER, NULL, 0);
    
    if (!wait_for_usb_ack(5000)) {
        printf("Failed to receive footer ACK\n");
        return false;
    }
    
    printf("USB transfer completed successfully!\n");
    return true;
}

// Main function for Pico B
int main() {
    stdio_init_all();
    sleep_ms(2000); // Wait for USB enumeration
    
    printf("Pico B: Image Receiver and USB Transfer\n");
    printf("Waiting for SPI data...\n");
    
    // Initialize SPI as slave
    spi_slave_init();
    
    while (1) {
        // Reset for new transfer
        transfer_complete = false;
        bytes_received = 0;
        image_size = 0;
        
        // Receive image via SPI
        process_spi_packets();
        
        if (transfer_complete && bytes_received > 0) {
            // Transfer to computer via USB
            transfer_image_to_computer(image_buffer, bytes_received);
        }
        
        sleep_ms(100);
    }
    
    return 0;
}
```

## Part 5: Computer Side Software

### Step 1: Install Python and Dependencies

**Windows:**
```bash
# Download Python from python.org
# During installation, check "Add Python to PATH"
pip install pyserial pillow
```

**Linux/Mac:**
```bash
sudo apt install python3 python3-pip  # Linux
# or
brew install python3  # Mac

pip3 install pyserial pillow
```

### Step 2: Create Computer Receiver Script

Create `computer_receiver.py`:
```python
#!/usr/bin/env python3
import serial
import serial.tools.list_ports
import time
import struct
import os
from datetime import datetime
from PIL import Image
import io

# Protocol constants matching Pico B
START_MARKER = 0x55
SYNC_BYTE = 0x3C
PACKET_HEADER = 0x01
PACKET_DATA = 0x02
PACKET_FOOTER = 0x03
ACK_BYTE = 0x06
NAK_BYTE = 0x15

class ImageReceiver:
    def __init__(self, port=None, output_dir="received_images"):
        self.port = port or self.find_pico_port()
        if not self.port:
            raise Exception("No Pico device found!")
        
        self.ser = serial.Serial(self.port, timeout=5.0)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Connected to {self.port}")
    
    def find_pico_port(self):
        """Auto-detect Pico port"""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if "Pico" in port.description or "2E8A" in port.hwid:
                return port.device
        return None
    
    def fletcher16(self, data):
        """Calculate Fletcher-16 checksum"""
        sum1 = sum2 = 0
        for byte in data:
            sum1 = (sum1 + byte) % 255
            sum2 = (sum2 + sum1) % 255
        return (sum2 << 8) | sum1
    
    def receive_packet(self):
        """Receive a packet from Pico"""
        # Look for start marker
        while True:
            byte = self.ser.read(1)
            if not byte:
                return None, None
            if byte[0] == START_MARKER:
                break
        
        # Read rest of header
        header = self.ser.read(3)
        if len(header) < 3:
            return None, None
        
        sync = header[0]
        packet_type = header[1]
        length = header[2]
        
        # Read data
        data = self.ser.read(length) if length > 0 else b''
        
        # Read checksum and end marker
        footer = self.ser.read(2)
        if len(footer) < 2:
            return None, None
        
        checksum = footer[0]
        end_marker = footer[1]
        
        # Verify packet
        if sync != SYNC_BYTE or end_marker != START_MARKER:
            return None, None
        
        # Verify checksum
        checksum_data = bytes([sync, packet_type, length]) + data
        calc_checksum = self.fletcher16(checksum_data) & 0xFF
        
        if calc_checksum != checksum:
            return None, None
        
        return packet_type, data
    
    def send_ack(self, success=True):
        """Send acknowledgment"""
        self.ser.write(bytes([ACK_BYTE if success else NAK_BYTE]))
        self.ser.flush()
    
    def receive_image(self):
        """Receive complete image"""
        print("Waiting for image...")
        
        image_data = bytearray()
        image_size = 0
        
        while True:
            packet_type, data = self.receive_packet()
            
            if packet_type is None:
                print("Timeout or error receiving packet")
                continue
            
            if packet_type == PACKET_HEADER:
                # Extract image size
                if len(data) >= 4:
                    image_size = struct.unpack('<I', data[:4])[0]
                    print(f"Receiving image: {image_size} bytes")
                    image_data = bytearray()
                    self.send_ack(True)
                else:
                    self.send_ack(False)
            
            elif packet_type == PACKET_DATA:
                # Append data
                image_data.extend(data)
                progress = (len(image_data) * 100) // image_size if image_size > 0 else 0
                print(f"Progress: {progress}% ({len(image_data)}/{image_size} bytes)", end='\r')
                self.send_ack(True)
            
            elif packet_type == PACKET_FOOTER:
                # Transfer complete
                print(f"\nTransfer complete: {len(image_data)} bytes")
                self.send_ack(True)
                
                if len(image_data) == image_size:
                    return bytes(image_data)
                else:
                    print(f"Size mismatch: expected {image_size}, got {len(image_data)}")
                    return None
    
    def save_image(self, image_data):
        """Save image to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Try to detect image format
        if image_data[:2] == b'\xff\xd8':
            # JPEG
            filename = f"image_{timestamp}.jpg"
        elif image_data[:2] == b'BM':
            # BMP
            filename = f"image_{timestamp}.bmp"
        else:
            # Unknown, save as binary
            filename = f"image_{timestamp}.bin"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        print(f"Saved image: {filepath}")
        
        # Try to open and display image info
        try:
            img = Image.open(io.BytesIO(image_data))
            print(f"Image info: {img.format} {img.size} {img.mode}")
        except:
            print("Could not parse image format")
        
        return filepath
    
    def run(self):
        """Main receiving loop"""
        print("Image receiver ready. Waiting for images...")
        
        try:
            while True:
                image_data = self.receive_image()
                if image_data:
                    self.save_image(image_data)
                    print("\nWaiting for next image...")
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.ser.close()

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    port = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Create and run receiver
    receiver = ImageReceiver(port)
    receiver.run()
```

### Step 3: Create GUI Application (Optional)

Create `image_receiver_gui.py`:
```python
#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
from PIL import Image, ImageTk
import os
from computer_receiver import ImageReceiver

class ImageReceiverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Microscope Image Receiver")
        self.root.geometry("800x600")
        
        # Message queue for thread communication
        self.message_queue = queue.Queue()
        
        # Setup GUI
        self.setup_gui()
        
        # Start receiver thread
        self.receiver_thread = None
        self.running = False
    
    def setup_gui(self):
        # Control frame
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        self.start_button = ttk.Button(control_frame, text="Start Receiving", 
                                      command=self.toggle_receiving)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.status_label = ttk.Label(control_frame, text="Status: Stopped")
        self.status_label.grid(row=0, column=1, padx=20)
        
        # Log frame
        log_frame = ttk.Frame(self.root, padding="10")
        log_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image preview frame
        preview_frame = ttk.Frame(self.root, padding="10")
        preview_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.image_label = ttk.Label(preview_frame, text="No image received")
        self.image_label.grid(row=0, column=0)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(2, weight=2)
        
        # Start message processor
        self.process_messages()
    
    def toggle_receiving(self):
        if not self.running:
            self.start_receiving()
        else:
            self.stop_receiving()
    
    def start_receiving(self):
        self.running = True
        self.start_button.config(text="Stop Receiving")
        self.status_label.config(text="Status: Running")
        
        # Start receiver thread
        self.receiver_thread = threading.Thread(target=self.receiver_worker)
        self.receiver_thread.daemon = True
        self.receiver_thread.start()
    
    def stop_receiving(self):
        self.running = False
        self.start_button.config(text="Start Receiving")
        self.status_label.config(text="Status: Stopped")
    
    def receiver_worker(self):
        """Worker thread for receiving images"""
        try:
            receiver = ImageReceiver()
            self.message_queue.put(("log", "Connected to Pico"))
            
            while self.running:
                image_data = receiver.receive_image()
                if image_data:
                    filepath = receiver.save_image(image_data)
                    self.message_queue.put(("image", filepath))
                    self.message_queue.put(("log", f"Received image: {filepath}"))
            
            receiver.ser.close()
            
        except Exception as e:
            self.message_queue.put(("error", str(e)))
            self.running = False
    
    def process_messages(self):
        """Process messages from worker thread"""
        try:
            while True:
                msg_type, msg_data = self.message_queue.get_nowait()
                
                if msg_type == "log":
                    self.log_text.insert(tk.END, msg_data + "\n")
                    self.log_text.see(tk.END)
                
                elif msg_type == "image":
                    self.display_image(msg_data)
                
                elif msg_type == "error":
                    self.log_text.insert(tk.END, f"ERROR: {msg_data}\n")
                    self.stop_receiving()
        
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_messages)
    
    def display_image(self, filepath):
        """Display image preview"""
        try:
            # Load and resize image
            img = Image.open(filepath)
            img.thumbnail((400, 300), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep reference
            
        except Exception as e:
            self.log_text.insert(tk.END, f"Failed to display image: {e}\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageReceiverGUI(root)
    root.mainloop()
```

## Part 6: Building and Flashing the Firmware

### Step 1: Build Pico A Firmware

```bash
cd pico_a
mkdir build
cd build
cmake ..
make

# This creates pico_a_sd_reader.uf2
```

### Step 2: Build Pico B Firmware

```bash
cd ../../pico_b
mkdir build
cd build
cmake ..
make

# This creates pico_b_receiver.uf2
```

### Step 3: Flash Firmware to Picos

**For Pico A:**
1. Hold BOOTSEL button on Pico A
2. Connect USB cable
3. Release BOOTSEL button
4. Pico appears as USB drive
5. Copy `pico_a_sd_reader.uf2` to the drive
6. Pico automatically reboots

**For Pico B:**
1. Repeat same process with Pico B
2. Copy `pico_b_receiver.uf2` to the drive

## Part 7: Complete System Testing

### Step 1: Hardware Setup Checklist

- [ ] SD card module connected to Pico A (check all 6 wires)
- [ ] Pico A and Pico B connected via SPI (4 wires + GND)
- [ ] Pico B connected to computer via USB
- [ ] SD card formatted as FAT32 and inserted

### Step 2: Test Procedure

1. **Power on sequence:**
   ```
   1. Connect Pico B to computer
   2. Start computer receiver script
   3. Power on Pico A
   4. Insert SD card into Dimension Scope
   ```

2. **Capture test image:**
   - Use Dimension Scope to capture image
   - Watch Pico A serial output for detection
   - Monitor transfer progress
   - Verify image received on computer

### Step 3: Troubleshooting Guide

**SD Card Not Detected:**
- Check 3.3V power to SD module
- Verify CS pull-up resistor
- Try different SD card
- Check with multimeter

**SPI Communication Fails:**
- Verify all SPI connections
- Check common ground
- Reduce SPI speed to 1MHz
- Add 10ms delays between packets

**USB Transfer Issues:**
- Install USB CDC drivers (Windows)
- Check device manager for COM port
- Try different USB cable
- Reduce packet size

## Alternative Approach: Direct Image Streaming

For simpler implementation without reconstruction, use this streaming approach:

```c
// Pico A: Stream image directly
void stream_image_direct(FileInfo *file) {
    uint32_t offset = 0;
    uint8_t buffer[512];
    
    while (offset < file->size) {
        uint32_t bytes_read;
        read_image_chunk(file, offset, buffer, &bytes_read, 512);
        
        // Send raw data to Pico B
        gpio_put(17, 0);
        spi_write_blocking(SPI_TRANSFER_PORT, buffer, bytes_read);
        gpio_put(17, 1);
        
        offset += bytes_read;
        sleep_ms(1); // Give receiver time
    }
}

// Pico B: Forward to USB immediately
void forward_to_usb_direct() {
    uint8_t buffer[512];
    
    while (1) {
        // Wait for SPI data
        if (spi_is_readable(SPI_TRANSFER_PORT)) {
            int bytes = spi_read_blocking(SPI_TRANSFER_PORT, 0xFF, buffer, 512);
            
            // Send directly to USB
            fwrite(buffer, 1, bytes, stdout);
            fflush(stdout);
        }
    }
}
```

This comprehensive guide provides everything needed to build a complete image transfer system from the Dimension Scope to a computer using two Raspberry Pi Pico 2 boards, with detailed code, wiring diagrams, and troubleshooting steps.