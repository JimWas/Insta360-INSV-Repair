#!/usr/bin/env python3
"""
Insta360 X4 .insv File Repair Tool
===================================
Repairs broken/corrupted .insv video files from the Insta360 X4 camera.

Common corruption scenarios:
  1. Missing moov atom (recording interrupted by power loss / crash)
  2. Truncated mdat (incomplete recording)
  3. Corrupted atom headers

Usage:
  python3 insv_repair.py <broken_file.insv> [--reference <good_file.insv>]
  python3 insv_repair.py <broken_file.insv> --scan   (no reference file needed)
  python3 insv_repair.py <file.insv> --diagnose       (analyze only, no repair)

The tool can repair files in two modes:
  - Reference mode: Uses a known-good .insv file from the same camera/settings
    to reconstruct the moov atom with correct codec parameters.
  - Scan mode: Scans the raw mdat data to detect HEVC/AAC frames and rebuilds
    the moov atom from scratch (slower but doesn't need a reference).
"""

import argparse
import os
import struct
import sys
import time
from collections import namedtuple
from typing import BinaryIO, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTAINER_ATOMS = {b'moov', b'trak', b'mdia', b'minf', b'stbl', b'dinf',
                   b'edts', b'udta', b'mvex', b'sinf', b'schi'}

# HEVC NAL unit types relevant for frame detection
HEVC_NAL_VPS = 32
HEVC_NAL_SPS = 33
HEVC_NAL_PPS = 34
HEVC_NAL_AUD = 35  # Access Unit Delimiter
HEVC_NAL_IDR_W_RADL = 19
HEVC_NAL_IDR_N_LP = 20
HEVC_NAL_CRA = 21
HEVC_NAL_TRAIL_N = 0
HEVC_NAL_TRAIL_R = 1
HEVC_NAL_TSA_N = 2
HEVC_NAL_TSA_R = 3
HEVC_NAL_STSA_N = 4
HEVC_NAL_STSA_R = 5
HEVC_NAL_RADL_N = 6
HEVC_NAL_RADL_R = 7
HEVC_NAL_RASL_N = 8
HEVC_NAL_RASL_R = 9
HEVC_NAL_BLA_W_LP = 16
HEVC_NAL_BLA_W_RADL = 17
HEVC_NAL_BLA_N_LP = 18
HEVC_NAL_SEI_PREFIX = 39
HEVC_NAL_SEI_SUFFIX = 40

HEVC_IDR_TYPES = {HEVC_NAL_IDR_W_RADL, HEVC_NAL_IDR_N_LP, HEVC_NAL_CRA,
                  HEVC_NAL_BLA_W_LP, HEVC_NAL_BLA_W_RADL, HEVC_NAL_BLA_N_LP}
HEVC_SLICE_TYPES = (set(range(0, 10)) | set(range(16, 22)))

# AAC ADTS sync word
AAC_SYNC = 0xFFF

# Insta360 X4 defaults
X4_VIDEO_WIDTH = 2880
X4_VIDEO_HEIGHT = 2880
X4_VIDEO_TIMESCALE = 60000
X4_AUDIO_TIMESCALE = 48000
X4_AUDIO_CHANNELS = 2
X4_AUDIO_SAMPLE_RATE = 48000

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

Atom = namedtuple('Atom', ['offset', 'size', 'type', 'header_size'])
FrameInfo = namedtuple('FrameInfo', ['offset', 'size', 'is_keyframe', 'track'])
SampleEntry = namedtuple('SampleEntry', ['offset', 'size', 'duration', 'is_sync'])


# ---------------------------------------------------------------------------
# MP4/MOV Atom Parser
# ---------------------------------------------------------------------------

class AtomParser:
    """Parse MP4/MOV atom (box) structure."""

    def __init__(self, f: BinaryIO, file_size: int):
        self.f = f
        self.file_size = file_size

    def read_atom_header(self, offset: int) -> Optional[Atom]:
        """Read an atom header at the given offset."""
        self.f.seek(offset)
        header = self.f.read(8)
        if len(header) < 8:
            return None
        size, atom_type = struct.unpack('>I4s', header)
        header_size = 8

        if size == 1:  # 64-bit extended size
            ext = self.f.read(8)
            if len(ext) < 8:
                return None
            size = struct.unpack('>Q', ext)[0]
            header_size = 16
        elif size == 0:  # extends to end of file
            size = self.file_size - offset

        return Atom(offset=offset, size=size, type=atom_type, header_size=header_size)

    def parse_top_level(self) -> list[Atom]:
        """Parse all top-level atoms."""
        atoms = []
        offset = 0
        while offset < self.file_size:
            atom = self.read_atom_header(offset)
            if atom is None or atom.size < 8:
                break
            atoms.append(atom)
            offset += atom.size
        return atoms

    def parse_children(self, parent: Atom) -> list[Atom]:
        """Parse child atoms within a container atom."""
        children = []
        offset = parent.offset + parent.header_size
        end = parent.offset + parent.size
        while offset < end:
            atom = self.read_atom_header(offset)
            if atom is None or atom.size < 8:
                break
            children.append(atom)
            offset += atom.size
        return children

    def find_atom(self, atoms: list[Atom], atom_type: bytes) -> Optional[Atom]:
        """Find first atom of given type in a list."""
        for a in atoms:
            if a.type == atom_type:
                return a
        return None

    def read_atom_data(self, atom: Atom) -> bytes:
        """Read the full data payload of an atom (excluding header)."""
        self.f.seek(atom.offset + atom.header_size)
        return self.f.read(atom.size - atom.header_size)

    def extract_atom_raw(self, atom: Atom) -> bytes:
        """Read the entire atom including header."""
        self.f.seek(atom.offset)
        return self.f.read(atom.size)


# ---------------------------------------------------------------------------
# HEVC Stream Scanner
# ---------------------------------------------------------------------------

class HEVCScanner:
    """Scan raw byte stream for HEVC NAL units and frame boundaries."""

    @staticmethod
    def get_nal_type(nal_header_byte: int) -> int:
        """Extract NAL unit type from first byte of NAL header."""
        return (nal_header_byte >> 1) & 0x3F

    @staticmethod
    def is_keyframe_nal(nal_type: int) -> bool:
        return nal_type in HEVC_IDR_TYPES

    @staticmethod
    def is_slice_nal(nal_type: int) -> bool:
        return nal_type in HEVC_SLICE_TYPES


# ---------------------------------------------------------------------------
# MP4 Length-Prefixed NAL Scanner
# ---------------------------------------------------------------------------

class MP4NalScanner:
    """
    Scan mdat that uses MP4-style length-prefixed NAL units.

    In MP4/MOV containers, HEVC data uses 4-byte big-endian length prefixes
    instead of Annex-B start codes (0x000001). Each video sample in the mdat
    consists of one or more length-prefixed NAL units.

    For Insta360 X4 files, the mdat contains interleaved chunks:
      - Video track 1 (lens 1) chunks
      - Video track 2 (lens 2) chunks
      - Audio chunks
    These are arranged in chunk groups, with the chunk offsets stored in
    the stco/co64 atoms of each track.
    """

    def __init__(self, f: BinaryIO, mdat_offset: int, mdat_size: int):
        self.f = f
        self.mdat_offset = mdat_offset
        self.mdat_data_offset = mdat_offset + 8  # skip mdat header
        self.mdat_size = mdat_size
        self.mdat_end = mdat_offset + mdat_size

    def scan_nal_units_at(self, offset: int, max_bytes: int = 1024 * 1024) -> list:
        """Scan length-prefixed NAL units starting at offset.

        Returns list of (offset, size, nal_type) tuples.
        """
        nals = []
        self.f.seek(offset)
        pos = offset
        end = min(offset + max_bytes, self.mdat_end)

        while pos < end:
            self.f.seek(pos)
            length_bytes = self.f.read(4)
            if len(length_bytes) < 4:
                break
            nal_length = struct.unpack('>I', length_bytes)[0]

            # Sanity check: NAL length should be reasonable
            if nal_length == 0 or nal_length > 50 * 1024 * 1024:
                break

            if pos + 4 + nal_length > self.mdat_end:
                break

            # Read NAL header (2 bytes for HEVC)
            nal_header = self.f.read(min(2, nal_length))
            if len(nal_header) < 1:
                break

            nal_type = HEVCScanner.get_nal_type(nal_header[0])
            nals.append((pos, 4 + nal_length, nal_type))
            pos += 4 + nal_length

        return nals

    def identify_frame_at(self, offset: int) -> Optional[dict]:
        """Try to identify what kind of frame/data starts at offset.

        Returns dict with keys: type ('video'|'audio'|'unknown'),
        size, is_keyframe, nal_types.
        """
        self.f.seek(offset)
        header = self.f.read(8)
        if len(header) < 8:
            return None

        # Try as length-prefixed HEVC NAL unit
        nal_length = struct.unpack('>I', header[:4])[0]
        if 1 <= nal_length <= 50 * 1024 * 1024:
            nal_type = HEVCScanner.get_nal_type(header[4])
            if nal_type <= 40:  # Valid HEVC NAL type range
                # Scan all NALs in this access unit
                nals = self.scan_nal_units_at(offset)
                if nals:
                    total_size = sum(n[1] for n in nals)
                    nal_types = [n[2] for n in nals]
                    is_keyframe = any(HEVCScanner.is_keyframe_nal(t) for t in nal_types)
                    has_slice = any(HEVCScanner.is_slice_nal(t) for t in nal_types)
                    if has_slice or is_keyframe:
                        return {
                            'type': 'video',
                            'size': total_size,
                            'is_keyframe': is_keyframe,
                            'nal_types': nal_types
                        }

        # Try as AAC ADTS frame
        if header[0] == 0xFF and (header[1] & 0xF0) == 0xF0:
            # ADTS header
            frame_length = ((header[3] & 0x03) << 11) | (header[4] << 3) | ((header[5] >> 5) & 0x07)
            if 7 <= frame_length <= 8192:
                return {
                    'type': 'audio',
                    'size': frame_length,
                    'is_keyframe': True,
                    'nal_types': []
                }

        return None


# ---------------------------------------------------------------------------
# Moov Reconstruction
# ---------------------------------------------------------------------------

class MoovBuilder:
    """Build a moov atom for a repaired file."""

    def __init__(self, timescale: int = 600):
        self.timescale = timescale

    @staticmethod
    def build_ftyp() -> bytes:
        """Build ftyp atom matching Insta360 X4 format."""
        major_brand = b'avc1'
        minor_version = struct.pack('>I', 0x20140200)
        compatible = b'avc1' + b'isom'
        data = major_brand + minor_version + compatible
        return struct.pack('>I', len(data) + 8) + b'ftyp' + data

    def build_mvhd(self, duration_ts: int, next_track_id: int = 4) -> bytes:
        """Build mvhd (movie header) atom."""
        # Version 0 mvhd
        data = bytearray()
        data += struct.pack('>I', 0)  # version + flags
        data += struct.pack('>I', 0)  # creation_time
        data += struct.pack('>I', 0)  # modification_time
        data += struct.pack('>I', self.timescale)  # timescale
        data += struct.pack('>I', duration_ts)  # duration
        data += struct.pack('>I', 0x00010000)  # rate (1.0)
        data += struct.pack('>H', 0x0100)  # volume (1.0)
        data += b'\x00' * 10  # reserved
        # Matrix (identity)
        data += struct.pack('>9I',
            0x00010000, 0, 0,
            0, 0x00010000, 0,
            0, 0, 0x40000000)
        data += b'\x00' * 24  # pre_defined
        data += struct.pack('>I', next_track_id)
        return struct.pack('>I', len(data) + 8) + b'mvhd' + bytes(data)

    def build_tkhd(self, track_id: int, duration_ts: int,
                   width: int = 0, height: int = 0, is_audio: bool = False) -> bytes:
        """Build tkhd (track header) atom."""
        data = bytearray()
        flags = 0x000003  # track_enabled | track_in_movie
        data += struct.pack('>I', flags)  # version 0 + flags
        data += struct.pack('>I', 0)  # creation_time
        data += struct.pack('>I', 0)  # modification_time
        data += struct.pack('>I', track_id)
        data += struct.pack('>I', 0)  # reserved
        data += struct.pack('>I', duration_ts)  # duration
        data += b'\x00' * 8  # reserved
        data += struct.pack('>H', 0)  # layer
        data += struct.pack('>H', 0)  # alternate_group
        data += struct.pack('>H', 0x0100 if is_audio else 0)  # volume
        data += b'\x00' * 2  # reserved
        # Matrix (identity)
        data += struct.pack('>9I',
            0x00010000, 0, 0,
            0, 0x00010000, 0,
            0, 0, 0x40000000)
        data += struct.pack('>I', width << 16)  # width (fixed-point)
        data += struct.pack('>I', height << 16)  # height (fixed-point)
        return struct.pack('>I', len(data) + 8) + b'tkhd' + bytes(data)

    def build_mdhd(self, timescale: int, duration: int) -> bytes:
        """Build mdhd (media header) atom."""
        data = bytearray()
        data += struct.pack('>I', 0)  # version + flags
        data += struct.pack('>I', 0)  # creation_time
        data += struct.pack('>I', 0)  # modification_time
        data += struct.pack('>I', timescale)
        data += struct.pack('>I', duration)
        data += struct.pack('>H', 0x55C4)  # language (undetermined)
        data += struct.pack('>H', 0)  # quality
        return struct.pack('>I', len(data) + 8) + b'mdhd' + bytes(data)

    def build_hdlr(self, handler_type: bytes, name: str) -> bytes:
        """Build hdlr (handler reference) atom."""
        data = bytearray()
        data += struct.pack('>I', 0)  # version + flags
        data += b'\x00' * 4  # pre_defined
        data += handler_type  # handler_type (4 bytes)
        data += b'\x00' * 12  # reserved
        data += name.encode('utf-8') + b'\x00'
        return struct.pack('>I', len(data) + 8) + b'hdlr' + bytes(data)

    def build_stsd_hevc(self, width: int, height: int,
                        vps: bytes, sps: bytes, pps: bytes) -> bytes:
        """Build stsd atom with hvc1 sample description for HEVC."""
        # hvcC configuration record
        hvcc = bytearray()
        hvcc += struct.pack('>B', 1)  # configurationVersion
        # general_profile_space(2) | general_tier_flag(1) | general_profile_idc(5)
        hvcc += struct.pack('>B', 0x01)  # Main profile
        hvcc += struct.pack('>I', 0x60000000)  # general_profile_compatibility_flags
        # general_constraint_indicator_flags (6 bytes)
        hvcc += b'\x90\x00\x00\x00\x00\x00'
        hvcc += struct.pack('>B', 153)  # general_level_idc (5.1 = 153)
        hvcc += struct.pack('>H', 0xF000)  # min_spatial_segmentation_idc
        hvcc += struct.pack('>B', 0xFC)  # parallelismType
        hvcc += struct.pack('>B', 0xFD)  # chromaFormat (1 = 4:2:0)
        hvcc += struct.pack('>B', 0xF8)  # bitDepthLumaMinus8
        hvcc += struct.pack('>B', 0xF8)  # bitDepthChromaMinus8
        hvcc += struct.pack('>H', 0)  # avgFrameRate
        hvcc += struct.pack('>B', 0x0F)  # constantFrameRate | numTemporalLayers | lengthSizeMinusOne (3 = 4 bytes)
        hvcc += struct.pack('>B', 3)  # numOfArrays (VPS, SPS, PPS)

        # VPS array
        hvcc += struct.pack('>B', 0xA0)  # array_completeness(1) | nal_unit_type(5) = 32 (VPS)
        hvcc += struct.pack('>H', 1)  # numNalus
        hvcc += struct.pack('>H', len(vps))
        hvcc += vps

        # SPS array
        hvcc += struct.pack('>B', 0xA1)  # nal_unit_type = 33 (SPS)
        hvcc += struct.pack('>H', 1)
        hvcc += struct.pack('>H', len(sps))
        hvcc += sps

        # PPS array
        hvcc += struct.pack('>B', 0xA2)  # nal_unit_type = 34 (PPS)
        hvcc += struct.pack('>H', 1)
        hvcc += struct.pack('>H', len(pps))
        hvcc += pps

        hvcc_atom = struct.pack('>I', len(hvcc) + 8) + b'hvcC' + bytes(hvcc)

        # hvc1 sample entry
        entry = bytearray()
        entry += b'\x00' * 6  # reserved
        entry += struct.pack('>H', 1)  # data_reference_index
        entry += struct.pack('>H', 0)  # pre_defined
        entry += struct.pack('>H', 0)  # reserved
        entry += b'\x00' * 12  # pre_defined
        entry += struct.pack('>H', width)
        entry += struct.pack('>H', height)
        entry += struct.pack('>I', 0x00480000)  # horizresolution (72 dpi)
        entry += struct.pack('>I', 0x00480000)  # vertresolution
        entry += struct.pack('>I', 0)  # reserved
        entry += struct.pack('>H', 1)  # frame_count
        entry += b'\x00' * 32  # compressorname
        entry += struct.pack('>H', 0x0018)  # depth
        entry += struct.pack('>h', -1)  # pre_defined
        entry += hvcc_atom
        entry_atom = struct.pack('>I', len(entry) + 8) + b'hvc1' + bytes(entry)

        # stsd
        stsd_data = struct.pack('>I', 0) + struct.pack('>I', 1) + entry_atom  # version+flags, entry_count
        return struct.pack('>I', len(stsd_data) + 8) + b'stsd' + stsd_data

    def build_stsd_aac(self, sample_rate: int = 48000, channels: int = 2) -> bytes:
        """Build stsd atom with mp4a sample description for AAC."""
        # esds atom (Elementary Stream Descriptor)
        # Simplified AAC-LC descriptor
        aac_config = self._build_aac_config(sample_rate, channels)
        esds = self._build_esds(aac_config)
        esds_atom = struct.pack('>I', len(esds) + 8) + b'esds' + esds

        # mp4a sample entry
        entry = bytearray()
        entry += b'\x00' * 6  # reserved
        entry += struct.pack('>H', 1)  # data_reference_index
        entry += b'\x00' * 8  # reserved
        entry += struct.pack('>H', channels)  # channel_count
        entry += struct.pack('>H', 16)  # sample_size (bits)
        entry += struct.pack('>H', 0)  # compression_id
        entry += struct.pack('>H', 0)  # packet_size
        entry += struct.pack('>I', sample_rate << 16)  # sample_rate (fixed-point)
        entry += esds_atom
        entry_atom = struct.pack('>I', len(entry) + 8) + b'mp4a' + bytes(entry)

        stsd_data = struct.pack('>I', 0) + struct.pack('>I', 1) + entry_atom
        return struct.pack('>I', len(stsd_data) + 8) + b'stsd' + stsd_data

    @staticmethod
    def _build_aac_config(sample_rate: int, channels: int) -> bytes:
        """Build AudioSpecificConfig for AAC-LC."""
        # AAC-LC object type = 2
        freq_index = {96000: 0, 88200: 1, 64000: 2, 48000: 3, 44100: 4,
                      32000: 5, 24000: 6, 22050: 7, 16000: 8, 12000: 9,
                      11025: 10, 8000: 11}.get(sample_rate, 3)
        # 5 bits object type + 4 bits freq index + 4 bits channel config + 3 bits frame length flag etc.
        config = ((2 << 11) | (freq_index << 7) | (channels << 3)) & 0xFFFF
        return struct.pack('>H', config)

    @staticmethod
    def _build_esds(aac_config: bytes) -> bytes:
        """Build esds (Elementary Stream Descriptor) data."""
        data = bytearray()
        data += struct.pack('>I', 0)  # version + flags

        # ES_Descriptor
        es_desc = bytearray()
        es_desc += struct.pack('>H', 2)  # ES_ID
        es_desc += struct.pack('>B', 0)  # streamDependenceFlag etc.

        # DecoderConfigDescriptor
        dec_config = bytearray()
        dec_config += struct.pack('>B', 0x40)  # objectTypeIndication (AAC)
        dec_config += struct.pack('>B', 0x15)  # streamType (audio)
        dec_config += b'\x00\x00\x00'  # bufferSizeDB
        dec_config += struct.pack('>I', 128000)  # maxBitrate
        dec_config += struct.pack('>I', 128000)  # avgBitrate

        # DecoderSpecificInfo
        dsi = bytes([0x05, len(aac_config)]) + aac_config
        dec_config += dsi

        dec_config_tagged = bytes([0x04, len(dec_config)]) + dec_config
        es_desc += dec_config_tagged

        # SLConfigDescriptor
        es_desc += bytes([0x06, 0x01, 0x02])

        es_desc_tagged = bytes([0x03, len(es_desc)]) + es_desc
        data += es_desc_tagged

        return bytes(data)

    def build_stts(self, entries: list[tuple[int, int]]) -> bytes:
        """Build stts (time-to-sample) atom.
        entries: list of (sample_count, sample_delta)
        """
        data = struct.pack('>I', 0)  # version + flags
        data += struct.pack('>I', len(entries))
        for count, delta in entries:
            data += struct.pack('>II', count, delta)
        return struct.pack('>I', len(data) + 8) + b'stts' + data

    def build_stss(self, sync_samples: list[int]) -> bytes:
        """Build stss (sync sample / keyframe table) atom."""
        data = struct.pack('>I', 0)
        data += struct.pack('>I', len(sync_samples))
        for s in sync_samples:
            data += struct.pack('>I', s)
        return struct.pack('>I', len(data) + 8) + b'stss' + data

    def build_stsc(self, entries: list[tuple[int, int, int]]) -> bytes:
        """Build stsc (sample-to-chunk) atom.
        entries: list of (first_chunk, samples_per_chunk, sample_desc_index)
        """
        data = struct.pack('>I', 0)
        data += struct.pack('>I', len(entries))
        for fc, spc, sdi in entries:
            data += struct.pack('>III', fc, spc, sdi)
        return struct.pack('>I', len(data) + 8) + b'stsc' + data

    def build_stsz(self, sample_sizes: list[int]) -> bytes:
        """Build stsz (sample size) atom."""
        data = struct.pack('>I', 0)  # version + flags
        data += struct.pack('>I', 0)  # sample_size (0 = variable)
        data += struct.pack('>I', len(sample_sizes))
        for s in sample_sizes:
            data += struct.pack('>I', s)
        return struct.pack('>I', len(data) + 8) + b'stsz' + data

    def build_co64(self, chunk_offsets: list[int]) -> bytes:
        """Build co64 (64-bit chunk offset) atom."""
        data = struct.pack('>I', 0)
        data += struct.pack('>I', len(chunk_offsets))
        for off in chunk_offsets:
            data += struct.pack('>Q', off)
        return struct.pack('>I', len(data) + 8) + b'co64' + data

    def build_stbl(self, stsd: bytes, stts: bytes, stsc: bytes,
                   stsz: bytes, co64: bytes, stss: bytes = None) -> bytes:
        """Build stbl (sample table) container."""
        children = stsd + stts + stsc + stsz + co64
        if stss:
            children += stss
        return struct.pack('>I', len(children) + 8) + b'stbl' + children

    def build_dinf(self) -> bytes:
        """Build dinf + dref atoms."""
        # dref with one self-referencing entry
        dref_entry = struct.pack('>I', 12) + b'url ' + struct.pack('>I', 1)
        dref_data = struct.pack('>I', 0) + struct.pack('>I', 1) + dref_entry
        dref = struct.pack('>I', len(dref_data) + 8) + b'dref' + dref_data
        return struct.pack('>I', len(dref) + 8) + b'dinf' + dref

    def build_vmhd(self) -> bytes:
        """Build vmhd (video media header)."""
        data = struct.pack('>I', 1)  # version 0, flags = 1
        data += struct.pack('>H', 0)  # graphicsmode
        data += struct.pack('>3H', 0, 0, 0)  # opcolor
        return struct.pack('>I', len(data) + 8) + b'vmhd' + data

    def build_smhd(self) -> bytes:
        """Build smhd (sound media header)."""
        data = struct.pack('>I', 0)  # version + flags
        data += struct.pack('>H', 0)  # balance
        data += struct.pack('>H', 0)  # reserved
        return struct.pack('>I', len(data) + 8) + b'smhd' + data

    def build_minf(self, media_header: bytes, dinf: bytes, stbl: bytes) -> bytes:
        """Build minf container."""
        children = media_header + dinf + stbl
        return struct.pack('>I', len(children) + 8) + b'minf' + children

    def build_mdia(self, mdhd: bytes, hdlr: bytes, minf: bytes) -> bytes:
        """Build mdia container."""
        children = mdhd + hdlr + minf
        return struct.pack('>I', len(children) + 8) + b'mdia' + children

    def build_trak(self, tkhd: bytes, mdia: bytes) -> bytes:
        """Build trak container."""
        children = tkhd + mdia
        return struct.pack('>I', len(children) + 8) + b'trak' + children

    def build_moov(self, mvhd: bytes, traks: list[bytes], udta: bytes = None) -> bytes:
        """Build moov container."""
        children = mvhd
        for trak in traks:
            children += trak
        if udta:
            children += udta
        return struct.pack('>I', len(children) + 8) + b'moov' + children


# ---------------------------------------------------------------------------
# Reference File Analyzer
# ---------------------------------------------------------------------------

class ReferenceAnalyzer:
    """Extract codec parameters and structure from a reference .insv file."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.f = open(filepath, 'rb')
        self.file_size = self.f.seek(0, 2)
        self.parser = AtomParser(self.f, self.file_size)
        self.tracks = []

    def close(self):
        self.f.close()

    def analyze(self) -> dict:
        """Analyze reference file and extract all needed parameters."""
        top_atoms = self.parser.parse_top_level()

        info = {
            'ftyp': None,
            'moov_offset': None,
            'mdat_offset': None,
            'mdat_size': None,
            'tracks': [],
            'mvhd_timescale': None,
            'udta': None,
        }

        for atom in top_atoms:
            if atom.type == b'ftyp':
                info['ftyp'] = self.parser.extract_atom_raw(atom)
            elif atom.type == b'mdat':
                info['mdat_offset'] = atom.offset
                info['mdat_size'] = atom.size
            elif atom.type == b'moov':
                info['moov_offset'] = atom.offset
                self._parse_moov(atom, info)

        return info

    def _parse_moov(self, moov: Atom, info: dict):
        """Parse moov atom and extract track information."""
        children = self.parser.parse_children(moov)

        for child in children:
            if child.type == b'mvhd':
                data = self.parser.read_atom_data(child)
                version = data[0]
                if version == 0:
                    info['mvhd_timescale'] = struct.unpack('>I', data[12:16])[0]
                else:
                    info['mvhd_timescale'] = struct.unpack('>I', data[20:24])[0]

            elif child.type == b'trak':
                track_info = self._parse_trak(child)
                if track_info:
                    info['tracks'].append(track_info)

            elif child.type == b'udta':
                info['udta'] = self.parser.extract_atom_raw(child)

    def _parse_trak(self, trak: Atom) -> Optional[dict]:
        """Parse a single track atom."""
        children = self.parser.parse_children(trak)
        track = {
            'track_id': None,
            'handler_type': None,
            'handler_name': '',
            'timescale': None,
            'codec': None,
            'width': 0,
            'height': 0,
            'sample_count': 0,
            'stsd_raw': None,
            'vps': None, 'sps': None, 'pps': None,
            'sample_delta': None,
            'samples_per_chunk': None,
            'chunk_count': 0,
        }

        for child in children:
            if child.type == b'tkhd':
                data = self.parser.read_atom_data(child)
                version = data[0]
                if version == 0:
                    track['track_id'] = struct.unpack('>I', data[12:16])[0]
                    track['width'] = struct.unpack('>I', data[76:80])[0] >> 16
                    track['height'] = struct.unpack('>I', data[80:84])[0] >> 16

            elif child.type == b'mdia':
                self._parse_mdia(child, track)

        return track

    def _parse_mdia(self, mdia: Atom, track: dict):
        """Parse mdia atom."""
        children = self.parser.parse_children(mdia)
        for child in children:
            if child.type == b'mdhd':
                data = self.parser.read_atom_data(child)
                version = data[0]
                if version == 0:
                    track['timescale'] = struct.unpack('>I', data[12:16])[0]
                else:
                    track['timescale'] = struct.unpack('>I', data[20:24])[0]

            elif child.type == b'hdlr':
                data = self.parser.read_atom_data(child)
                track['handler_type'] = data[8:12]
                track['handler_name'] = data[24:].decode('ascii', errors='replace').rstrip('\x00')

            elif child.type == b'minf':
                self._parse_minf(child, track)

    def _parse_minf(self, minf: Atom, track: dict):
        """Parse minf atom."""
        children = self.parser.parse_children(minf)
        for child in children:
            if child.type == b'stbl':
                self._parse_stbl(child, track)

    def _parse_stbl(self, stbl: Atom, track: dict):
        """Parse stbl atom and extract codec/sample info."""
        children = self.parser.parse_children(stbl)
        for child in children:
            if child.type == b'stsd':
                data = self.parser.read_atom_data(child)
                track['stsd_raw'] = self.parser.extract_atom_raw(child)
                # Parse codec type from first sample entry
                entry_type = data[12:16]
                track['codec'] = entry_type.decode('ascii', errors='replace')

                # Extract HEVC parameters if applicable
                if entry_type == b'hvc1' or entry_type == b'hev1':
                    self._extract_hevc_params(data[8:], track)

            elif child.type == b'stts':
                data = self.parser.read_atom_data(child)
                entry_count = struct.unpack('>I', data[4:8])[0]
                if entry_count > 0:
                    count, delta = struct.unpack('>II', data[8:16])
                    track['sample_delta'] = delta

            elif child.type == b'stsz':
                data = self.parser.read_atom_data(child)
                track['sample_count'] = struct.unpack('>I', data[8:12])[0]

            elif child.type == b'stsc':
                data = self.parser.read_atom_data(child)
                entry_count = struct.unpack('>I', data[4:8])[0]
                if entry_count > 0:
                    _, spc, _ = struct.unpack('>III', data[8:20])
                    track['samples_per_chunk'] = spc

            elif child.type == b'co64':
                data = self.parser.read_atom_data(child)
                track['chunk_count'] = struct.unpack('>I', data[4:8])[0]

            elif child.type == b'stco':
                data = self.parser.read_atom_data(child)
                track['chunk_count'] = struct.unpack('>I', data[4:8])[0]

    def _extract_hevc_params(self, entry_data: bytes, track: dict):
        """Extract VPS, SPS, PPS from hvc1 sample entry."""
        # Navigate to hvcC box within the sample entry
        # Sample entry: size(4) type(4) reserved(6) dri(2) ...
        # visual: + pre_defined(2) reserved(2) pre_defined2(12) w(2) h(2)
        #         + hres(4) vres(4) reserved(4) frame_count(2)
        #         + compressorname(32) depth(2) pre_defined(2)
        # Then child boxes follow
        offset = 4 + 6 + 2 + 2 + 2 + 12 + 2 + 2 + 4 + 4 + 4 + 2 + 32 + 2 + 2
        entry_size = struct.unpack('>I', entry_data[:4])[0]

        while offset < entry_size - 8:
            box_size = struct.unpack('>I', entry_data[offset:offset + 4])[0]
            box_type = entry_data[offset + 4:offset + 8]
            if box_size < 8:
                break
            if box_type == b'hvcC':
                hvcc_data = entry_data[offset + 8:offset + box_size]
                self._parse_hvcc(hvcc_data, track)
                break
            offset += box_size

    @staticmethod
    def _parse_hvcc(data: bytes, track: dict):
        """Parse HEVCDecoderConfigurationRecord to extract VPS/SPS/PPS."""
        if len(data) < 23:
            return
        num_arrays = data[22]
        offset = 23
        for _ in range(num_arrays):
            if offset + 3 > len(data):
                break
            nal_type = data[offset] & 0x3F
            num_nalus = struct.unpack('>H', data[offset + 1:offset + 3])[0]
            offset += 3
            for _ in range(num_nalus):
                if offset + 2 > len(data):
                    break
                nal_length = struct.unpack('>H', data[offset:offset + 2])[0]
                offset += 2
                if offset + nal_length > len(data):
                    break
                nal_data = data[offset:offset + nal_length]
                if nal_type == HEVC_NAL_VPS:
                    track['vps'] = nal_data
                elif nal_type == HEVC_NAL_SPS:
                    track['sps'] = nal_data
                elif nal_type == HEVC_NAL_PPS:
                    track['pps'] = nal_data
                offset += nal_length


# ---------------------------------------------------------------------------
# MDAT Chunk/Sample Scanner
# ---------------------------------------------------------------------------

class MdatScanner:
    """Scan mdat to discover interleaved chunks and build sample tables.

    Insta360 X4 mdat layout (all chunks contiguous, zero gaps):
      - Video samples: length-prefixed HEVC NAL units (4-byte BE length + NAL data)
      - Audio samples: raw AAC frames (~505 bytes, NO ADTS headers)
      - Interleave: V1, V2 alternate; audio chunks appear periodically between them
      - Each chunk = 1 sample (samples_per_chunk = 1 for all tracks)
    """

    def __init__(self, f: BinaryIO, mdat_offset: int, mdat_size: int,
                 mdat_header_size: int = 8):
        self.f = f
        self.mdat_offset = mdat_offset
        self.mdat_header_size = mdat_header_size
        self.mdat_data_start = mdat_offset + mdat_header_size
        self.mdat_end = mdat_offset + mdat_size
        self.mdat_size = mdat_size

    def _try_parse_hevc_sample(self, pos: int) -> Optional[tuple]:
        """Try to parse one HEVC video sample (access unit) at pos.

        Returns (total_size, is_keyframe, nal_types) or None if not valid HEVC.
        """
        if pos + 6 > self.mdat_end:
            return None

        self.f.seek(pos)
        first_4 = self.f.read(4)
        if len(first_4) < 4:
            return None
        first_len = struct.unpack('>I', first_4)[0]

        # Quick validation: length must be reasonable
        if first_len < 2 or first_len > 30 * 1024 * 1024:
            return None

        # Read first NAL header
        nal_hdr = self.f.read(2)
        if len(nal_hdr) < 2:
            return None
        first_nal_type = HEVCScanner.get_nal_type(nal_hdr[0])

        # Must be a valid HEVC NAL type
        if first_nal_type > 40:
            return None

        # HEVC NAL header has forbidden_zero_bit=0 at MSB
        if nal_hdr[0] & 0x80:
            return None

        # Now consume all NAL units in this access unit
        total_size = 0
        is_keyframe = False
        nal_types = []
        cur = pos
        found_slice = False

        while cur + 4 < self.mdat_end:
            self.f.seek(cur)
            lb = self.f.read(4)
            if len(lb) < 4:
                break
            nal_length = struct.unpack('>I', lb)[0]

            if nal_length < 2 or nal_length > 30 * 1024 * 1024:
                break
            if cur + 4 + nal_length > self.mdat_end:
                break

            nh = self.f.read(2)
            if len(nh) < 2:
                break

            # Validate NAL header: forbidden_zero_bit must be 0
            if nh[0] & 0x80:
                break

            nal_type = HEVCScanner.get_nal_type(nh[0])
            if nal_type > 40:
                break

            nal_types.append(nal_type)

            if nal_type in HEVC_IDR_TYPES:
                is_keyframe = True

            total_size += 4 + nal_length
            cur += 4 + nal_length

            # Parameter sets / SEI: continue to next NAL in same sample
            if nal_type in (HEVC_NAL_VPS, HEVC_NAL_SPS, HEVC_NAL_PPS,
                           HEVC_NAL_AUD, HEVC_NAL_SEI_PREFIX, HEVC_NAL_SEI_SUFFIX):
                continue

            if nal_type in HEVC_SLICE_TYPES or nal_type in HEVC_IDR_TYPES:
                found_slice = True
                # Peek at next NAL to see if it's still part of this picture
                if cur + 6 <= self.mdat_end:
                    self.f.seek(cur)
                    peek = self.f.read(6)
                    if len(peek) >= 6:
                        peek_len = struct.unpack('>I', peek[:4])[0]
                        if 2 <= peek_len <= 30 * 1024 * 1024:
                            peek_type = HEVCScanner.get_nal_type(peek[4])
                            # If next is SEI suffix, include it
                            if peek_type == HEVC_NAL_SEI_SUFFIX:
                                continue
                            # If next is same slice type without first_slice flag, include it
                            if peek_type in HEVC_SLICE_TYPES and not (peek[5] & 0x80):
                                continue
                break

        if total_size == 0 or not found_slice:
            return None

        # Additional validation: total size should be > 100 bytes for a real frame
        if total_size < 100:
            return None

        return (total_size, is_keyframe, nal_types)

    def _find_next_hevc_start(self, start: int, max_search: int = 64 * 1024) -> Optional[int]:
        """Find the next position that looks like a valid HEVC sample start.

        Used to determine audio chunk boundaries (audio = gap between video chunks).
        """
        # Read a buffer and scan for valid HEVC NAL length + header patterns
        search_end = min(start + max_search, self.mdat_end)
        pos = start

        while pos < search_end:
            result = self._try_parse_hevc_sample(pos)
            if result is not None:
                return pos
            pos += 1  # Byte-by-byte search (audio chunks are small ~505 bytes)

        return None

    def scan_chunks_with_reference(self, ref_info: dict) -> dict:
        """Scan mdat using reference file parameters to rebuild sample tables.

        Walks sequentially through contiguous mdat data. At each position:
        - Try to parse as HEVC video sample
        - If not HEVC, search ahead for next HEVC start (gap = audio data)
        """
        tracks = ref_info['tracks']
        num_video_tracks = sum(1 for t in tracks if t.get('handler_type') == b'vide')
        has_audio = any(t.get('handler_type') == b'soun' for t in tracks)

        # Map track indices
        video_track_indices = [i for i, t in enumerate(tracks) if t.get('handler_type') == b'vide']
        audio_track_idx = next((i for i, t in enumerate(tracks) if t.get('handler_type') == b'soun'), None)

        result = {}
        for i, track in enumerate(tracks):
            result[i] = {
                'chunk_offsets': [],
                'sample_sizes': [],
                'sync_samples': [],
                'samples_per_chunk': 1,
                'timescale': track.get('timescale', 60000),
                'sample_delta': track.get('sample_delta', 1001),
            }

        pos = self.mdat_data_start
        total_bytes = self.mdat_end - self.mdat_data_start
        last_progress = -1
        video_sample_count = {i: 0 for i in video_track_indices}
        audio_sample_count = 0
        next_video_track = 0  # Alternates between 0 and 1

        print(f"  Scanning {total_bytes / (1024*1024*1024):.2f} GB of media data...")

        while pos < self.mdat_end - 6:
            # Progress reporting
            progress = int((pos - self.mdat_data_start) / total_bytes * 100)
            if progress >= last_progress + 5:
                print(f"  Progress: {progress}% ({(pos - self.mdat_data_start) / (1024*1024*1024):.2f} GB)")
                last_progress = progress

            # Try to parse as HEVC video sample
            hevc_result = self._try_parse_hevc_sample(pos)

            if hevc_result is not None:
                sample_size, is_keyframe, nal_types = hevc_result

                # Assign to alternating video track
                track_idx = video_track_indices[next_video_track % num_video_tracks]
                next_video_track += 1

                result[track_idx]['chunk_offsets'].append(pos)
                result[track_idx]['sample_sizes'].append(sample_size)
                video_sample_count[track_idx] = video_sample_count.get(track_idx, 0) + 1

                if is_keyframe:
                    result[track_idx]['sync_samples'].append(
                        video_sample_count[track_idx])

                pos += sample_size
            else:
                # Not HEVC - this is audio data (raw AAC, no ADTS headers)
                if audio_track_idx is not None and has_audio:
                    # Find where the next video sample starts
                    next_hevc = self._find_next_hevc_start(pos + 1)

                    if next_hevc is not None:
                        audio_region = next_hevc - pos

                        # Audio frames in X4 are ~505 bytes each
                        # Split the region into individual audio samples
                        audio_pos = pos
                        while audio_pos + 100 <= next_hevc:
                            # Determine this audio frame's size
                            remaining = next_hevc - audio_pos
                            # Try to find frame boundary by checking if a valid
                            # HEVC NAL starts after ~505 bytes
                            frame_size = min(505, remaining)

                            # For more accuracy, check nearby offsets
                            if remaining > 505:
                                # See if 505 bytes brings us to another audio frame
                                # or to a HEVC start
                                check = self._try_parse_hevc_sample(audio_pos + 505)
                                if check is not None:
                                    frame_size = 505
                                elif remaining >= 1010:
                                    # Might be multiple audio frames
                                    frame_size = 505
                                else:
                                    frame_size = remaining

                            result[audio_track_idx]['chunk_offsets'].append(audio_pos)
                            result[audio_track_idx]['sample_sizes'].append(frame_size)
                            audio_sample_count += 1
                            audio_pos += frame_size

                        pos = next_hevc
                    else:
                        # No more HEVC found - remaining data might be audio
                        remaining = self.mdat_end - pos
                        if remaining > 100:
                            audio_pos = pos
                            while audio_pos + 100 < self.mdat_end:
                                frame_size = min(505, self.mdat_end - audio_pos)
                                if frame_size < 50:
                                    break
                                result[audio_track_idx]['chunk_offsets'].append(audio_pos)
                                result[audio_track_idx]['sample_sizes'].append(frame_size)
                                audio_sample_count += 1
                                audio_pos += frame_size
                        break
                else:
                    # No audio track - try to skip and find next video
                    next_hevc = self._find_next_hevc_start(pos + 1, max_search=1024*1024)
                    if next_hevc:
                        pos = next_hevc
                    else:
                        break

        # Print summary
        for i, track in enumerate(tracks):
            handler = track.get('handler_type', b'????')
            n_samples = len(result[i]['sample_sizes'])
            n_chunks = len(result[i]['chunk_offsets'])
            n_sync = len(result[i]['sync_samples'])
            total_data = sum(result[i]['sample_sizes'])
            print(f"  Track {i+1} ({handler.decode('ascii', errors='replace')}): "
                  f"{n_samples} samples, {n_chunks} chunks, {n_sync} keyframes, "
                  f"{total_data/(1024*1024):.1f} MB")

        return result

    def scan_mdat_standalone(self) -> dict:
        """Scan mdat without a reference file.

        Detects HEVC video frames and raw AAC audio by walking sequentially.
        Assumes Insta360 X4 dual-lens layout.
        """
        print("  Performing standalone mdat scan (no reference file)...")
        print("  This may take a while for large files...")

        result = {
            0: {'chunk_offsets': [], 'sample_sizes': [], 'sync_samples': [],
                'samples_per_chunk': 1, 'timescale': X4_VIDEO_TIMESCALE,
                'sample_delta': 1001},
            1: {'chunk_offsets': [], 'sample_sizes': [], 'sync_samples': [],
                'samples_per_chunk': 1, 'timescale': X4_VIDEO_TIMESCALE,
                'sample_delta': 1001},
            2: {'chunk_offsets': [], 'sample_sizes': [], 'sync_samples': [],
                'samples_per_chunk': 1, 'timescale': X4_AUDIO_TIMESCALE,
                'sample_delta': 1024},
        }

        pos = self.mdat_data_start
        total_bytes = self.mdat_end - self.mdat_data_start
        last_progress = -1
        video_count = {0: 0, 1: 0}
        audio_count = 0
        next_video_track = 0

        while pos < self.mdat_end - 6:
            progress = int((pos - self.mdat_data_start) / total_bytes * 100)
            if progress >= last_progress + 5:
                print(f"  Progress: {progress}%")
                last_progress = progress

            hevc_result = self._try_parse_hevc_sample(pos)

            if hevc_result is not None:
                sample_size, is_keyframe, nal_types = hevc_result
                track_idx = next_video_track % 2
                next_video_track += 1

                result[track_idx]['chunk_offsets'].append(pos)
                result[track_idx]['sample_sizes'].append(sample_size)
                video_count[track_idx] += 1
                if is_keyframe:
                    result[track_idx]['sync_samples'].append(video_count[track_idx])

                pos += sample_size
            else:
                # Audio data - find next HEVC start
                next_hevc = self._find_next_hevc_start(pos + 1)
                if next_hevc is not None:
                    audio_pos = pos
                    while audio_pos + 50 <= next_hevc:
                        frame_size = min(505, next_hevc - audio_pos)
                        if frame_size < 50:
                            break
                        result[2]['chunk_offsets'].append(audio_pos)
                        result[2]['sample_sizes'].append(frame_size)
                        audio_count += 1
                        audio_pos += frame_size
                    pos = next_hevc
                else:
                    # Try larger search
                    next_hevc = self._find_next_hevc_start(pos + 1, max_search=1024 * 1024)
                    if next_hevc:
                        pos = next_hevc
                    else:
                        break

        for i in range(3):
            n = len(result[i]['sample_sizes'])
            names = ['Video 1', 'Video 2', 'Audio']
            print(f"  {names[i]}: {n} samples, {len(result[i]['chunk_offsets'])} chunks")

        return result


# ---------------------------------------------------------------------------
# Diagnoser
# ---------------------------------------------------------------------------

class INSVDiagnoser:
    """Diagnose issues with an .insv file."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.f = open(filepath, 'rb')
        self.file_size = self.f.seek(0, 2)
        self.parser = AtomParser(self.f, self.file_size)

    def close(self):
        self.f.close()

    def diagnose(self) -> dict:
        """Run full diagnosis and return findings."""
        findings = {
            'file_size': self.file_size,
            'has_ftyp': False,
            'has_mdat': False,
            'has_moov': False,
            'mdat_offset': None,
            'mdat_size': None,
            'moov_offset': None,
            'moov_size': None,
            'issues': [],
            'tracks': [],
            'repairable': False,
            'repair_strategy': None,
        }

        print(f"\n{'='*60}")
        print(f"INSV File Diagnosis")
        print(f"{'='*60}")
        print(f"File: {self.filepath}")
        print(f"Size: {self.file_size:,} bytes ({self.file_size / (1024*1024*1024):.2f} GB)")
        print()

        # Parse top-level atoms
        atoms = self.parser.parse_top_level()
        print("Top-level atoms:")
        for atom in atoms:
            type_str = atom.type.decode('ascii', errors='replace')
            print(f"  {atom.offset:#014x}  {type_str:6s}  {atom.size:>14,} bytes")

        print()

        # Check for required atoms
        for atom in atoms:
            if atom.type == b'ftyp':
                findings['has_ftyp'] = True
                self.f.seek(atom.offset + 8)
                brand = self.f.read(4)
                print(f"  ftyp brand: {brand.decode('ascii', errors='replace')}")
            elif atom.type == b'mdat':
                findings['has_mdat'] = True
                findings['mdat_offset'] = atom.offset
                findings['mdat_size'] = atom.size
            elif atom.type == b'moov':
                findings['has_moov'] = True
                findings['moov_offset'] = atom.offset
                findings['moov_size'] = atom.size

        # Diagnose issues
        if not findings['has_ftyp']:
            findings['issues'].append("CRITICAL: Missing ftyp atom (file type header)")

        if not findings['has_mdat']:
            findings['issues'].append("CRITICAL: Missing mdat atom (no media data)")
        else:
            # Check if mdat extends beyond file
            mdat_end = findings['mdat_offset'] + findings['mdat_size']
            if mdat_end > self.file_size:
                findings['issues'].append(
                    f"WARNING: mdat declares size {findings['mdat_size']:,} bytes "
                    f"but file is only {self.file_size:,} bytes "
                    f"(truncated by {mdat_end - self.file_size:,} bytes)")

        if not findings['has_moov']:
            findings['issues'].append("CRITICAL: Missing moov atom (no track/sample index)")
            findings['repairable'] = True
            findings['repair_strategy'] = 'rebuild_moov'

            # Try to detect if this is an Insta360 X4 file by scanning mdat start
            if findings['has_mdat']:
                self._probe_mdat(findings)

        else:
            # Moov exists - check its integrity
            moov_atom = self.parser.find_atom(atoms, b'moov')
            if moov_atom:
                self._check_moov(moov_atom, findings)

        # Print findings
        print("\nDiagnosis:")
        if findings['issues']:
            for issue in findings['issues']:
                print(f"  * {issue}")
        else:
            print("  No issues detected - file appears intact.")

        if findings['repairable']:
            print(f"\n  Repair strategy: {findings['repair_strategy']}")
            if findings['repair_strategy'] == 'rebuild_moov':
                print("  Use --reference <good_file.insv> for best results,")
                print("  or --scan to rebuild without a reference (slower).")

        print()
        return findings

    def _probe_mdat(self, findings: dict):
        """Probe the start of mdat to identify the camera/format."""
        mdat_data_start = findings['mdat_offset'] + 8
        self.f.seek(mdat_data_start)
        header = self.f.read(64)

        if len(header) < 8:
            return

        # Check for HEVC NAL units (length-prefixed)
        nal_length = struct.unpack('>I', header[:4])[0]
        if 1 <= nal_length <= 10 * 1024 * 1024:
            nal_type = HEVCScanner.get_nal_type(header[4])
            findings['detected_codec'] = f'HEVC (NAL type {nal_type})'
            if nal_type in (HEVC_NAL_VPS, HEVC_NAL_SPS, HEVC_NAL_PPS):
                findings['issues'].append(
                    f"INFO: mdat starts with HEVC parameter set (NAL {nal_type}) - "
                    "likely a keyframe at start, good for recovery")
            elif nal_type in HEVC_IDR_TYPES:
                findings['issues'].append(
                    "INFO: mdat starts with HEVC IDR frame - good for recovery")

    def _check_moov(self, moov: Atom, findings: dict):
        """Check moov atom integrity."""
        try:
            children = self.parser.parse_children(moov)
            track_count = sum(1 for c in children if c.type == b'trak')
            findings['issues'].append(f"INFO: moov contains {track_count} tracks")

            for child in children:
                if child.type == b'trak':
                    self._check_track(child, findings)

        except Exception as e:
            findings['issues'].append(f"ERROR: Failed to parse moov: {e}")
            findings['repairable'] = True
            findings['repair_strategy'] = 'rebuild_moov'

    def _check_track(self, trak: Atom, findings: dict):
        """Check track integrity."""
        try:
            children = self.parser.parse_children(trak)
            tkhd = self.parser.find_atom(children, b'tkhd')
            if tkhd:
                data = self.parser.read_atom_data(tkhd)
                track_id = struct.unpack('>I', data[12:16])[0]
                findings['tracks'].append({'track_id': track_id, 'ok': True})
        except Exception as e:
            findings['issues'].append(f"WARNING: Corrupted track: {e}")


# ---------------------------------------------------------------------------
# Repairer
# ---------------------------------------------------------------------------

class INSVRepairer:
    """Main repair logic for broken .insv files."""

    def __init__(self, broken_path: str, output_path: str = None):
        self.broken_path = broken_path
        self.output_path = output_path or self._default_output_path()

    def _default_output_path(self) -> str:
        base, ext = os.path.splitext(self.broken_path)
        return f"{base}_repaired{ext}"

    def repair_with_reference(self, reference_path: str):
        """Repair using a known-good reference file."""
        print(f"\n{'='*60}")
        print(f"INSV Repair (Reference Mode)")
        print(f"{'='*60}")
        print(f"Broken file:    {self.broken_path}")
        print(f"Reference file: {reference_path}")
        print(f"Output file:    {self.output_path}")
        print()

        # Analyze reference
        print("[1/4] Analyzing reference file...")
        ref = ReferenceAnalyzer(reference_path)
        ref_info = ref.analyze()
        ref.close()

        print(f"  Reference has {len(ref_info['tracks'])} tracks:")
        for i, t in enumerate(ref_info['tracks']):
            print(f"    Track {i+1}: {t['codec']} "
                  f"({t['handler_type'].decode('ascii', errors='replace')}) "
                  f"{t['width']}x{t['height']} "
                  f"timescale={t['timescale']} "
                  f"samples_per_chunk={t['samples_per_chunk']}")

        # Open broken file and find mdat
        print("\n[2/4] Analyzing broken file...")
        with open(self.broken_path, 'rb') as bf:
            bf_size = bf.seek(0, 2)
            parser = AtomParser(bf, bf_size)
            atoms = parser.parse_top_level()

            mdat = parser.find_atom(atoms, b'mdat')
            moov = parser.find_atom(atoms, b'moov')
            ftyp = parser.find_atom(atoms, b'ftyp')

            if not mdat:
                # Entire file might be mdat without proper header
                print("  No mdat atom found - treating entire file as media data")
                mdat_offset = 0
                mdat_size = bf_size
                mdat_header_size = 0
                has_ftyp = False
            else:
                mdat_offset = mdat.offset
                mdat_size = mdat.size
                mdat_header_size = mdat.header_size
                has_ftyp = ftyp is not None
                if mdat_size > bf_size - mdat_offset:
                    actual_mdat_size = bf_size - mdat_offset
                    print(f"  mdat truncated: declared {mdat_size:,} bytes, "
                          f"actual {actual_mdat_size:,} bytes")
                    mdat_size = actual_mdat_size

            print(f"  mdat at offset {mdat_offset:#x}, size {mdat_size:,} bytes")

            if moov:
                print(f"  WARNING: File already has a moov atom at {moov.offset:#x}")
                print(f"  Will rebuild moov anyway (existing one may be corrupted)")

            # Scan mdat
            print("\n[3/4] Scanning media data...")
            scanner = MdatScanner(bf, mdat_offset, mdat_size, mdat_header_size)
            scan_result = scanner.scan_chunks_with_reference(ref_info)

            if not scan_result or all(len(scan_result[i]['sample_sizes']) == 0
                                     for i in scan_result):
                print("\n  ERROR: No valid media data found in mdat")
                return False

            # Build repaired file
            print("\n[4/4] Building repaired file...")
            self._build_repaired_file(bf, ref_info, scan_result,
                                      mdat_offset, mdat_size,
                                      has_ftyp, ftyp)

        print(f"\nRepair complete: {self.output_path}")
        return True

    def repair_standalone(self):
        """Repair without a reference file by scanning mdat."""
        print(f"\n{'='*60}")
        print(f"INSV Repair (Scan Mode - No Reference)")
        print(f"{'='*60}")
        print(f"Broken file: {self.broken_path}")
        print(f"Output file: {self.output_path}")
        print()

        with open(self.broken_path, 'rb') as bf:
            bf_size = bf.seek(0, 2)
            parser = AtomParser(bf, bf_size)
            atoms = parser.parse_top_level()

            mdat = parser.find_atom(atoms, b'mdat')
            ftyp = parser.find_atom(atoms, b'ftyp')

            if not mdat:
                # Check if file starts with ftyp + raw data
                if ftyp:
                    mdat_offset = ftyp.offset + ftyp.size
                    mdat_size = bf_size - mdat_offset
                    mdat_header_size = 0
                else:
                    mdat_offset = 0
                    mdat_size = bf_size
                    mdat_header_size = 0
                print(f"  No mdat found, treating offset {mdat_offset:#x} to EOF as media data")
            else:
                mdat_offset = mdat.offset
                mdat_size = min(mdat.size, bf_size - mdat.offset)
                mdat_header_size = mdat.header_size

            print(f"  Media data: {mdat_size:,} bytes at offset {mdat_offset:#x}")

            # Scan
            print("\nScanning media data for frames...")
            scanner = MdatScanner(bf, mdat_offset, mdat_size, mdat_header_size)
            scan_result = scanner.scan_mdat_standalone()

            if not scan_result:
                print("\nERROR: Could not detect any valid frames")
                return False

            # Extract VPS/SPS/PPS from first keyframe
            vps, sps, pps = self._extract_hevc_params_from_mdat(bf, scan_result)
            if not all([vps, sps, pps]):
                print("WARNING: Could not extract HEVC parameters from media data")
                print("  Using default parameters (may not work with all players)")
                vps = vps or bytes([0x40, 0x01, 0x0C, 0x01, 0xFF, 0xFF, 0x01, 0x60,
                                    0x00, 0x00, 0x03, 0x00, 0x90, 0x00, 0x00, 0x03,
                                    0x00, 0x00, 0x03, 0x00, 0x99, 0x95, 0x98, 0x09])
                sps = sps or bytes([0x42, 0x01, 0x01, 0x01, 0x60, 0x00, 0x00, 0x03,
                                    0x00, 0x90, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03,
                                    0x00, 0x99, 0xA0, 0x01, 0xE0, 0x20])
                pps = pps or bytes([0x44, 0x01, 0xC0, 0xF7, 0xC0, 0xCC, 0x90])

            # Build reference-like info from scan results
            ref_info = self._build_ref_info_from_scan(scan_result, vps, sps, pps)

            # Build repaired file
            print("\nBuilding repaired file...")
            has_ftyp = ftyp is not None
            self._build_repaired_file(bf, ref_info, scan_result,
                                      mdat_offset, mdat_size,
                                      has_ftyp, ftyp)

        print(f"\nRepair complete: {self.output_path}")
        return True

    def _extract_hevc_params_from_mdat(self, f: BinaryIO, scan_result: dict) -> tuple:
        """Extract VPS, SPS, PPS from the first keyframe in mdat."""
        vps = sps = pps = None

        for track_idx in (0, 1):
            if track_idx not in scan_result:
                continue
            track = scan_result[track_idx]
            if not track['chunk_offsets']:
                continue

            # Check first few chunks for parameter sets
            for chunk_off in track['chunk_offsets'][:10]:
                f.seek(chunk_off)
                pos = chunk_off

                for _ in range(20):  # Max NAL units to check
                    f.seek(pos)
                    lb = f.read(4)
                    if len(lb) < 4:
                        break
                    nal_len = struct.unpack('>I', lb)[0]
                    if nal_len < 2 or nal_len > 20 * 1024 * 1024:
                        break

                    nal_data = f.read(nal_len)
                    if len(nal_data) < 2:
                        break

                    nal_type = HEVCScanner.get_nal_type(nal_data[0])
                    if nal_type == HEVC_NAL_VPS:
                        vps = nal_data
                    elif nal_type == HEVC_NAL_SPS:
                        sps = nal_data
                    elif nal_type == HEVC_NAL_PPS:
                        pps = nal_data

                    pos += 4 + nal_len

                    if vps and sps and pps:
                        return vps, sps, pps

        return vps, sps, pps

    def _build_ref_info_from_scan(self, scan_result: dict,
                                   vps: bytes, sps: bytes, pps: bytes) -> dict:
        """Build a reference info dict from scan results."""
        tracks = []
        for i in range(3):
            if i not in scan_result:
                continue
            if i < 2:  # Video tracks
                tracks.append({
                    'track_id': i + 1,
                    'handler_type': b'vide',
                    'handler_name': 'INS.HVC',
                    'timescale': X4_VIDEO_TIMESCALE,
                    'codec': 'hvc1',
                    'width': X4_VIDEO_WIDTH,
                    'height': X4_VIDEO_HEIGHT,
                    'sample_delta': 1001,
                    'samples_per_chunk': scan_result[i].get('samples_per_chunk', 1),
                    'vps': vps, 'sps': sps, 'pps': pps,
                })
            else:  # Audio
                tracks.append({
                    'track_id': i + 1,
                    'handler_type': b'soun',
                    'handler_name': 'INS.AAC',
                    'timescale': X4_AUDIO_TIMESCALE,
                    'codec': 'mp4a',
                    'width': 0, 'height': 0,
                    'sample_delta': 1024,
                    'samples_per_chunk': scan_result[i].get('samples_per_chunk', 1),
                })

        return {
            'tracks': tracks,
            'mvhd_timescale': 600,
            'udta': None,
            'ftyp': None,
        }

    def _build_repaired_file(self, broken_f: BinaryIO, ref_info: dict,
                              scan_result: dict, mdat_offset: int, mdat_size: int,
                              has_ftyp: bool, ftyp_atom: Atom = None):
        """Build the repaired output file."""
        builder = MoovBuilder(timescale=ref_info.get('mvhd_timescale', 600))
        tracks = ref_info['tracks']

        # Calculate the offset shift: in the repaired file, we write
        # ftyp first, then mdat, then moov
        # The mdat content stays the same but its absolute position changes

        # Build ftyp
        if has_ftyp and ftyp_atom:
            broken_f.seek(ftyp_atom.offset)
            ftyp_data = broken_f.read(ftyp_atom.size)
        else:
            ftyp_data = builder.build_ftyp()

        ftyp_size = len(ftyp_data)

        # Calculate new mdat offset and the shift for chunk offsets
        new_mdat_offset = ftyp_size
        old_mdat_data_start = mdat_offset + 8  # original mdat header is 8 bytes
        new_mdat_data_start = new_mdat_offset + 8
        offset_shift = new_mdat_data_start - old_mdat_data_start

        # Build track atoms
        trak_atoms = []
        max_duration_s = 0

        for i, track in enumerate(tracks):
            if i not in scan_result or not scan_result[i]['sample_sizes']:
                continue

            sr = scan_result[i]
            is_video = track.get('handler_type') == b'vide'
            timescale = sr.get('timescale', track.get('timescale', 60000))
            sample_delta = sr.get('sample_delta', track.get('sample_delta', 1001))
            n_samples = len(sr['sample_sizes'])
            duration_ts = n_samples * sample_delta
            duration_s = duration_ts / timescale
            max_duration_s = max(max_duration_s, duration_s)

            print(f"  Track {i+1}: {n_samples} samples, "
                  f"duration {duration_s:.1f}s")

            # Build stsd
            if is_video:
                vps = track.get('vps', b'')
                sps = track.get('sps', b'')
                pps = track.get('pps', b'')
                width = track.get('width', X4_VIDEO_WIDTH)
                height = track.get('height', X4_VIDEO_HEIGHT)

                if track.get('stsd_raw'):
                    # Use reference stsd directly
                    stsd = track['stsd_raw']
                else:
                    stsd = builder.build_stsd_hevc(width, height, vps, sps, pps)
            else:
                if track.get('stsd_raw'):
                    stsd = track['stsd_raw']
                else:
                    stsd = builder.build_stsd_aac(
                        sample_rate=X4_AUDIO_SAMPLE_RATE,
                        channels=X4_AUDIO_CHANNELS)
                width = 0
                height = 0

            # stts
            stts = builder.build_stts([(n_samples, sample_delta)])

            # stss (sync samples - video only)
            stss = None
            if is_video and sr['sync_samples']:
                stss = builder.build_stss(sr['sync_samples'])

            # stsz
            stsz = builder.build_stsz(sr['sample_sizes'])

            # stsc - build proper sample-to-chunk table
            stsc_entries = self._build_stsc_entries(sr)
            stsc = builder.build_stsc(stsc_entries)

            # co64 - adjust chunk offsets
            adjusted_offsets = [off + offset_shift for off in sr['chunk_offsets']]
            co64 = builder.build_co64(adjusted_offsets)

            # Assemble track
            stbl = builder.build_stbl(stsd, stts, stsc, stsz, co64, stss)

            if is_video:
                media_header = builder.build_vmhd()
            else:
                media_header = builder.build_smhd()

            dinf = builder.build_dinf()
            minf = builder.build_minf(media_header, dinf, stbl)

            handler_type = track.get('handler_type', b'vide')
            handler_name = track.get('handler_name', 'VideoHandler')
            hdlr = builder.build_hdlr(handler_type, handler_name)
            mdhd = builder.build_mdhd(timescale, duration_ts)
            mdia = builder.build_mdia(mdhd, hdlr, minf)

            mvhd_ts = ref_info.get('mvhd_timescale', 600)
            tkhd_duration = int(duration_s * mvhd_ts)
            tkhd = builder.build_tkhd(
                track_id=track.get('track_id', i + 1),
                duration_ts=tkhd_duration,
                width=width, height=height,
                is_audio=not is_video)
            trak_atom = builder.build_trak(tkhd, mdia)
            trak_atoms.append(trak_atom)

        # mvhd
        mvhd_ts = ref_info.get('mvhd_timescale', 600)
        mvhd_duration = int(max_duration_s * mvhd_ts)
        mvhd = builder.build_mvhd(mvhd_duration, next_track_id=len(trak_atoms) + 1)

        # udta (optional - from reference)
        udta = ref_info.get('udta')

        # Build moov
        moov = builder.build_moov(mvhd, trak_atoms, udta)

        # Write output file
        print(f"\n  Writing repaired file...")
        print(f"    ftyp: {len(ftyp_data):,} bytes")
        print(f"    mdat: {mdat_size:,} bytes")
        print(f"    moov: {len(moov):,} bytes")

        with open(self.output_path, 'wb') as out:
            # Write ftyp
            out.write(ftyp_data)

            # Write mdat (copy from broken file)
            broken_f.seek(mdat_offset)
            mdat_header = struct.pack('>I', 1) + b'mdat' + struct.pack('>Q', mdat_size)
            if mdat_size > 0xFFFFFFFF:
                # Use 64-bit mdat size
                out.write(mdat_header)
                # Skip original mdat header and copy data
                broken_f.seek(mdat_offset + 8)
                remaining = mdat_size - 8
            else:
                # Copy mdat as-is
                remaining = mdat_size

            # Buffered copy
            broken_f.seek(mdat_offset + (16 if mdat_size > 0xFFFFFFFF else 0))
            buf_size = 64 * 1024 * 1024  # 64MB buffer
            copied = 0
            if mdat_size > 0xFFFFFFFF:
                remaining = mdat_size - 16
                broken_f.seek(mdat_offset + 8)
            else:
                # Write original mdat header
                broken_f.seek(mdat_offset)
                mdat_hdr = broken_f.read(8)
                out.write(mdat_hdr)
                remaining = mdat_size - 8

            while remaining > 0:
                chunk = broken_f.read(min(buf_size, remaining))
                if not chunk:
                    break
                out.write(chunk)
                remaining -= len(chunk)
                copied += len(chunk)

            # Write moov
            out.write(moov)

        total_size = os.path.getsize(self.output_path)
        print(f"    Total: {total_size:,} bytes ({total_size / (1024*1024*1024):.2f} GB)")

    def _build_stsc_entries(self, sr: dict) -> list[tuple[int, int, int]]:
        """Build sample-to-chunk entries from scan results."""
        chunk_offsets = sr['chunk_offsets']
        sample_sizes = sr['sample_sizes']
        spc = sr.get('samples_per_chunk', 1)

        if not chunk_offsets:
            return [(1, 1, 1)]

        # Simple case: all chunks have same number of samples
        n_samples = len(sample_sizes)
        n_chunks = len(chunk_offsets)
        if n_chunks > 0 and n_samples == n_chunks * spc:
            return [(1, spc, 1)]

        # If each chunk = 1 sample (standalone scan mode)
        if n_chunks == n_samples:
            return [(1, 1, 1)]

        # Variable: calculate per-chunk sample counts
        # For now, assume uniform
        if n_chunks > 0:
            avg_spc = max(1, n_samples // n_chunks)
            return [(1, avg_spc, 1)]

        return [(1, 1, 1)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Insta360 X4 .insv File Repair Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diagnose a file (no repair):
  python3 insv_repair.py broken.insv --diagnose

  # Repair using a reference file (recommended):
  python3 insv_repair.py broken.insv --reference good_file.insv

  # Repair without a reference file (scan mode):
  python3 insv_repair.py broken.insv --scan

  # Specify output filename:
  python3 insv_repair.py broken.insv --scan -o repaired.insv
""")

    parser.add_argument('input', help='Path to the broken .insv file')
    parser.add_argument('--reference', '-r', help='Path to a known-good .insv reference file')
    parser.add_argument('--scan', '-s', action='store_true',
                        help='Scan mode: rebuild moov by scanning mdat (no reference needed)')
    parser.add_argument('--diagnose', '-d', action='store_true',
                        help='Diagnose only - do not repair')
    parser.add_argument('--output', '-o', help='Output file path')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: File not found: {args.input}")
        sys.exit(1)

    if args.diagnose:
        diag = INSVDiagnoser(args.input)
        diag.diagnose()
        diag.close()
        return

    if args.reference:
        if not os.path.exists(args.reference):
            print(f"Error: Reference file not found: {args.reference}")
            sys.exit(1)
        repairer = INSVRepairer(args.input, args.output)
        success = repairer.repair_with_reference(args.reference)
    elif args.scan:
        repairer = INSVRepairer(args.input, args.output)
        success = repairer.repair_standalone()
    else:
        # Default: diagnose first
        print("No repair mode specified. Running diagnosis...")
        print("Use --reference or --scan to repair.\n")
        diag = INSVDiagnoser(args.input)
        findings = diag.diagnose()
        diag.close()

        if findings['repairable']:
            print("This file can be repaired. Run with:")
            print(f"  python3 insv_repair.py {args.input} --scan")
            print(f"  python3 insv_repair.py {args.input} --reference <good_file.insv>")
        return

    if not success:
        print("\nRepair failed.")
        sys.exit(1)

    # Verify with ffprobe if available
    print("\nVerifying repaired file...")
    try:
        import subprocess
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries',
             'stream=index,codec_name,codec_type,width,height,duration,nb_frames',
             '-of', 'compact', repairer.output_path],
            capture_output=True, text=True, timeout=30)
        if result.stdout:
            print("  ffprobe output:")
            for line in result.stdout.strip().split('\n'):
                print(f"    {line}")
        if result.stderr:
            print(f"  ffprobe errors: {result.stderr.strip()}")
    except FileNotFoundError:
        print("  ffprobe not found - install ffmpeg to verify output")
    except Exception as e:
        print(f"  Verification error: {e}")


if __name__ == '__main__':
    main()
