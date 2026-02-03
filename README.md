# Insta360 X4 .insv File Repair Tool

Repairs broken or corrupted `.insv` video files from the Insta360 X4 camera.

## Common Corruption Scenarios

- **Missing moov atom** — recording interrupted by power loss or crash
- **Truncated mdat** — incomplete recording
- **Corrupted atom headers**

## Requirements

- Python 3.10+
- No external dependencies (stdlib only)
- Optional: [FFmpeg](https://ffmpeg.org/) (`ffprobe`) for output verification

## Usage

### Diagnose a file (no repair)

```bash
python3 insv_repair.py broken.insv --diagnose
```

### Repair using a reference file (recommended)

Uses a known-good `.insv` file recorded with the same camera and settings to reconstruct the moov atom with correct codec parameters.

```bash
python3 insv_repair.py broken.insv --reference good_file.insv
```

### Repair without a reference file (scan mode)

Scans the raw mdat data to detect HEVC/AAC frames and rebuilds the moov atom from scratch. Slower, but doesn't require a reference file.

```bash
python3 insv_repair.py broken.insv --scan
```

### Specify output filename

```bash
python3 insv_repair.py broken.insv --scan -o repaired.insv
```

By default, the output is written to `<original_name>_repaired.insv`.

## How It Works

1. **Diagnosis** — Parses the MP4/MOV atom structure to identify missing or corrupted atoms (ftyp, mdat, moov).
2. **mdat Scanning** — Walks through the media data sequentially, identifying HEVC video samples (length-prefixed NAL units) and raw AAC audio chunks. Insta360 X4 files interleave two video tracks (one per lens) with an audio track.
3. **moov Reconstruction** — Builds a complete moov atom with proper track headers, sample tables (stts, stss, stsc, stsz, co64), and codec configuration (hvcC for HEVC, esds for AAC).
4. **Output** — Writes a new file with ftyp + mdat + moov in the correct order.

## Limitations

- Designed specifically for the Insta360 X4 (dual-lens HEVC at 2880x2880, 48kHz stereo AAC). May work with other Insta360 models but is untested.
- Scan mode assumes ~505-byte raw AAC audio frames and alternating video track chunks. Files with non-standard interleaving patterns may produce incorrect results.
- Cannot recover data that was never written to disk. If recording was interrupted, only the data present in the file can be salvaged.

## License

MIT
