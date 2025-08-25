from typing import List, Union

import torch

from . import utils as struct_utils


class Curves:
    """
    This class provides functions for working with batches of line based
    curves with varying numbers of lines and points, and converting between
    representations.

    Within Curves, there are three different representations of the faces and
    verts data:

    List
      - only used for input as a starting point to convert to other representations.
    Padded
      - has specific batch dimension.
    Packed
      - no batch dimension.
      - has auxiliary variables used to index into the padded representation.
    
    """
    def __init__(
        self,
        points: Union[List[torch.Tensor], torch.Tensor, None],
        pad_value: Union[float, int, None] = 0.0,
    ) -> None:
        """
        Args:
            points:
                Can be either

                - List where each element is a tensor of shape (num_points, 3)
                  containing the (x, y, z) coordinates of each points.
                - Padded float tensor with shape (num_meshes, max_num_points, 3).
                  Meshes should be padded with fill value of 0 so they all have
                  the same number of points.
        """
        self._points_list = None
        self._points_packed = None  # (sum(P_n), 3)
        self._points_padded = None  # (N, max(P_n), 3)
        
        self._lines_packed = None  # (sum(L_n), 2)
        
        self._num_points_per_curve = None       # N
        self._curve_to_points_packed_first_idx = None    # N
        self._points_packed_to_curve_idx = None # sum(P_n)
        self._pad_value = pad_value
        
        self._N = 0  # batch size (number of meshes)
        self._P = 0  # (max) number of points per curve
        self.device = torch.device("cpu")
        
        if isinstance(points, (list, tuple)):
            self._points_list = points
            self._N = len(self._points_list)
            self._P = max([p.shape[0] for p in self._points_list])
            self.device = points[0].device
        elif torch.is_tensor(points):
            self._points_padded = points
            self._N = self._points_padded.shape[0]
            self._P = self._points_padded.shape[1]
            self.device = points.device
        elif points is not None:
            raise ValueError(
                "Points must be either a list or a tensor with \
                    shape (batch_size, N, 3) where N is the number of points."
            )

    def __len__(self) -> int:
        return self._N
    
    def from_packed(self,
                    points_packed: torch.Tensor,
                    split_size: list,
        ):
        """Initialize from packed representation.

        Args:
            points_packed (torch.Tensor): tensor of points of shape (sum(P_n), 3).
            split_size (list): list of number of points per curve.
        """
        self._points_packed = points_packed
        self._num_points_per_curve = split_size
        self._curve_to_points_packed_first_idx = torch.cat([
            torch.zeros(1, dtype=torch.int64, device=self.device),
            torch.cumsum(torch.tensor(split_size[:-1]), dim=0),
        ], dim=0)
        self._points_packed_to_curve_idx = torch.cat([
            torch.zeros(split_size[i], dtype=torch.int64, device=self.device) + i
            for i in range(len(split_size))
        ], dim=0)
        
        self._N = len(split_size)
        self._P = max(split_size)
        self.device = points_packed.device
        
        return self
    
    def points_list(self):
        """
        Get the list representation of the vertices.

        Returns:
            list of tensors of vertices of shape (P_n, 3).
        """
        if self._points_list is None:
            assert (
                self._points_padded is not None or self._points_packed is not None
            ), "points_padded or points_packed is required to compute points_list."
            
            if self._points_packed is not None:
                self._points_list = struct_utils.packed_to_list(
                    self._points_packed, self._num_points_per_curve.tolist()
                )
            elif self._points_padded is not None:
                split_size = [(p.reshape((p.shape[0], -1)) != self._pad_value
                               ).sum(dim=-1, dtype=torch.bool).sum().item()
                              for p in self._points_padded]
                self._points_list = struct_utils.padded_to_list(
                    self._points_padded, split_size
                )
        return self._points_list
        
    def points_packed(self):
        """
        Get the packed representation of the points.

        Returns:
            tensor of points of shape (sum(P_n), 3).
        """
        if self._points_packed is None:
            point_list_to_packed = struct_utils.list_to_packed(
                self.points_list()
            )
            self._points_packed = point_list_to_packed[0]
            self._num_points_per_curve = point_list_to_packed[1]
            self._curve_to_points_packed_first_idx = point_list_to_packed[2]
            self._points_packed_to_curve_idx = point_list_to_packed[3]
        
        return self._points_packed
    
    def points_padded(self):
        """
        Get the padded representation of the points.

        Returns:
            tensor of points of shape (N, max(P_n), 3).
        """
        if self._points_padded is None:
            assert self._pad_value is not None, "pad_value must be specified."
            self._points_padded = struct_utils.list_to_padded(
                self.points_list(), pad_value=self._pad_value
            )
        return self._points_padded
    
    def lines_packed(self):
        """
        Get the packed representation of the lines.

        Returns:
            tensor of lines of shape (sum(L_n), 2).
        """
        if self._lines_packed is None:
            num_points = self.points_packed().shape[0]
            curve_to_points_end_idxs = \
                self.curve_to_points_packed_first_idx()[range(self._N)] - 1
                
            mask = torch.ones(num_points, dtype=torch.bool, device=self.device)
            mask[curve_to_points_end_idxs] = False
            curve_to_lines_first_idx = torch.arange(num_points, device=self.device)[mask]
            
            self._lines_packed = torch.stack([
                curve_to_lines_first_idx,
                curve_to_lines_first_idx + 1,
            ], dim=1)
            
        return self._lines_packed
    
    def num_points_per_curve(self):
        """
        Get the number of points per curve.

        Returns:
            tensor of shape (N,) where N is the number of curves.
        """
        if self._num_points_per_curve is None:
            self.points_packed()
        return self._num_points_per_curve
    
    def curve_to_points_packed_first_idx(self):
        """
        Get the index of the first point for each curve in the packed representation.

        Returns:
            tensor of shape (N,) where N is the number of curves.
        """
        if self._curve_to_points_packed_first_idx is None:
            self.points_packed()
        return self._curve_to_points_packed_first_idx
    
    def points_packed_to_curve_idx(self):
        """
        Get the index of the curve for each point in the packed representation.

        Returns:
            tensor of shape (sum(P_n),) where P_n is the number of points in
            each curve.
        """
        if self._points_packed_to_curve_idx is None:
            self.points_packed()
        return self._points_packed_to_curve_idx
    
    def update_packed(self, new_points_packed):
        self.points_packed()
        self._points_packed = new_points_packed
        self._points_list = None
        self._points_padded = None
        
        return self
    
    def copy(self):
        """
        Shallow copy of Curves object.
        """
        new = Curves(self.points_list(), pad_value=self._pad_value)
        new._points_packed = self._points_packed
        new._points_padded = self._points_padded
        new._lines_packed = self._lines_packed
        new._num_points_per_curve = self._num_points_per_curve
        new._curve_to_points_packed_first_idx = \
            self._curve_to_points_packed_first_idx
        new._points_packed_to_curve_idx = self._points_packed_to_curve_idx
        
        return new