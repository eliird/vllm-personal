// Phase 2 memory-heavy CRIU probe.
// Allocates N MiB of anonymous memory, fills it with a known pattern, embeds a
// monotonically increasing counter in the region, and writes
// "<counter> <pattern_checksum>" to a status file each second.
//
// Build: gcc -O2 scripts/p2_ram.c -o scripts/p2_ram
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
    size_t mb = (argc > 1) ? strtoul(argv[1], 0, 10) : 1024;
    const char *status = (argc > 2) ? argv[2] : "/tmp/p2_state";
    size_t bytes = mb << 20;
    uint32_t *buf = malloc(bytes);
    if (!buf) { perror("malloc"); return 1; }
    for (size_t i = 0; i < bytes / 4; i++) buf[i] = (uint32_t)(i * 2654435761u);
    uint64_t counter = 0;
    for (;;) {
        buf[0] = (uint32_t)counter;           // counter embedded in the region
        uint64_t sum = 0;
        // Skip buf[0]: it holds the counter, so including it would make the
        // checksum track the counter. buf[1..] must be byte-identical across
        // checkpoint/restore.
        for (size_t i = 1; i < bytes / 4; i++) sum += buf[i];
        FILE *f = fopen(status, "w");
        if (f) {
            fprintf(f, "%llu %llu\n", (unsigned long long)counter,
                    (unsigned long long)sum);
            fclose(f);
        }
        counter++;
        sleep(1);
    }
}
