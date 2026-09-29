# Reference decoder and analysis harness (minimp3, pinned, unmodified).
CC      ?= gcc
CFLAGS  ?= -O2 -ffp-contract=off -fno-fast-math -Wall -Wno-unused-function
BUILD   := build
MINIMP3 := third_party/minimp3/minimp3.h

all: $(BUILD)/refdec $(BUILD)/refdec_f32 $(BUILD)/libharness_s16.so $(BUILD)/libharness_f32.so

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/refdec: csrc/refdec.c $(MINIMP3) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $< -lm

$(BUILD)/refdec_f32: csrc/refdec.c $(MINIMP3) | $(BUILD)
	$(CC) $(CFLAGS) -DMINIMP3_FLOAT_OUTPUT -o $@ $< -lm

$(BUILD)/libharness_s16.so: csrc/harness.c $(MINIMP3) | $(BUILD)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $< -lm

$(BUILD)/libharness_f32.so: csrc/harness.c $(MINIMP3) | $(BUILD)
	$(CC) $(CFLAGS) -DMINIMP3_FLOAT_OUTPUT -fPIC -shared -o $@ $< -lm

clean:
	rm -rf $(BUILD)

.PHONY: all clean
