"""This module implements utility functions for loading and saving meshes."""

import os
import struct
import numpy as np
import torch


def load_hair(filepath: str, device="cpu"):
    """Imports a cyHair format hair file.

    """
    with open(filepath, "rb") as fp:
        header = fp.read(4)
        assert header == b"HAIR"
        num_strands = struct.unpack("I", fp.read(4))[0]
        num_points = struct.unpack("I", fp.read(4))[0]
        bit_flags = int.from_bytes(fp.read(4), "little")
        fp.seek(40-16, os.SEEK_CUR)  # skips unused fields
        info = fp.read(88).decode("ascii").rstrip('\x00')

        seg_counts = np.frombuffer(fp.read(2 * num_strands), dtype=np.uint16)
        positions = np.frombuffer(fp.read(12 * num_points), dtype=np.float32).reshape(-1, 3)
        thicknesses = np.frombuffer(fp.read(4 * num_points), dtype=np.float32) if bit_flags & 0b00100 else\
                      np.empty(0)

    positions = torch.tensor(positions, device=device)
    point_list = torch.split(positions, (seg_counts + 1).tolist())
    return list(point_list)
    

def save_hair(filepath, num_verts, positions, thicknesses=None, info=""):
    """Saves the given strands to Cem Yuksel's hair format.

    The file format specification can be found in:
    http://www.cemyuksel.com/research/hairmodels/

    Args:
        filepath: path of the saved hair file.
        num_verts: list of point counts of each spline.
        positions: list of 3D point coordinates (tuples of 3 floats).
        thicknesses: list of point thicknesses.
        info: information string to be saved in the file.
    """
    if not filepath.endswith(".hair"):
        raise NotImplementedError(f"{filepath} is not a .hair file!")
    
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().numpy()
    if isinstance(thicknesses, torch.Tensor):
        thicknesses = thicknesses.detach().cpu().numpy()
    
    num_strands = len(num_verts)
    num_points = len(positions)

    print(f"Writing {num_strands} strands, {num_points} total points.")

    with open(filepath, "wb") as fp:
        fp.write(b"HAIR")
        fp.write(struct.pack("I", num_strands))
        fp.write(struct.pack("I", num_points))
        bit_flags = "11000" if thicknesses is None else "11100"
        fp.write(struct.pack("i", int(bit_flags[::-1], 2)))
        fp.write(b"\0" * (40-16)) # skips unused fields
        
        # file information
        fp.write(bytes((info + "\0"*88)[:88], "ascii"))

        fp.write(np.array(num_verts).astype(np.uint16)-1)
        fp.write(np.array(positions).astype(np.float32))
        if not thicknesses is None:
            fp.write(np.array(thicknesses).astype(np.float32))
