"""Chip geometry for any loaded architecture, and SANA-FE's message routes."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CoreInfo:
    tile_id: int
    offset: int
    core_id: int
    name: str
    units: tuple

    @property
    def key(self):
        return f'{self.tile_id}.{self.offset}'


@dataclass(frozen=True)
class TileInfo:
    tile_id: int
    x: int
    y: int
    name: str
    cores: tuple


@dataclass(frozen=True)
class ChipLayout:
    width: int
    height: int
    tiles: tuple

    @classmethod
    def from_chip(cls, chip):
        described = chip.describe()
        width = int(described['noc_width_in_tiles'])
        height = int(described['noc_height_in_tiles'])
        if width * height != len(described['tiles']):
            raise ValueError(f'{len(described["tiles"])} tiles do not fill a '
                             f'{width} x {height} mesh')
        tiles = []
        for index, tile in enumerate(described['tiles']):
            tile_id = int(tile['id'])
            if tile_id != index:
                raise ValueError(f'tile {index} reports id {tile_id}')
            x, y = divmod(tile_id, height)  # src/arch.cpp: x = id / height
            cores = tuple(
                CoreInfo(tile_id, offset, int(core['id']), core['name'],
                         tuple(unit['name'] for unit in core['pipeline_units']))
                for offset, core in enumerate(tile['cores']))
            tiles.append(TileInfo(tile_id, x, y, tile['name'], cores))
        return cls(width, height, tuple(tiles))

    def tile_at(self, x, y):
        return self.tiles[x * self.height + y]

    def core(self, key):
        tile, offset = (int(part) for part in key.split('.'))
        return self.tiles[tile].cores[offset]

    def to_dict(self):
        return {'width': self.width, 'height': self.height, 'tiles': [
            {'id': tile.tile_id, 'x': tile.x, 'y': tile.y, 'name': tile.name,
             'cores': [{'key': core.key, 'core_id': core.core_id,
                        'name': core.name, 'units': list(core.units)}
                       for core in tile.cores]}
            for tile in self.tiles]}


def xy_path(layout, src_tile, dst_tile):
    """Tiles visited by SANA-FE's dimension-order route, source and destination included.

    Mirrors src/schedule.cpp: all x steps at the source row, then all y steps
    at the destination column. The model records no per-router events, so any
    position drawn between routers is a reconstruction.
    """
    x, y = layout.tiles[src_tile].x, layout.tiles[src_tile].y
    dest_x, dest_y = layout.tiles[dst_tile].x, layout.tiles[dst_tile].y
    path = [src_tile]
    while x != dest_x:
        x += 1 if dest_x > x else -1
        path.append(layout.tile_at(x, y).tile_id)
    while y != dest_y:
        y += 1 if dest_y > y else -1
        path.append(layout.tile_at(x, y).tile_id)
    return path
