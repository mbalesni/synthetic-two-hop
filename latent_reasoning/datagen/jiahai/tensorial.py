from itertools import product
from collections import defaultdict, namedtuple

def long_to_tensorial(long, coords, value):
    '''
    Args:
        long - List[Dict[str, Any]] - long format
    '''

def flatten_data_array_coords_and_values(da):
    import pandas as pd
    # Get all dimension names
    dims = da.dims
    
    def convert_index_to_list(index):
        # index is either MultiIndex or Index
        assert isinstance(index, pd.Index)
        if isinstance(index, pd.MultiIndex):
            idx = namedtuple(index.name, index.names)
            return [idx(*tup) for tup in index.values]
        else:
            return list(index.values)
    # Create a list of coordinate values for each dimension
    coord_values = [convert_index_to_list(da.indexes[dim]) for dim in dims]
    
    flattened_coords = list(product(*coord_values))
    
    # Flatten the data values
    flattened_values = da.values.flatten()
    
    # Combine coordinates and values
    result = [
        {
            **{
                cname: cvalue
                for cname, cvalue in zip(dims, coords)
            },
            da.name: value
        }
        for coords, value in zip(flattened_coords, flattened_values, strict=True)
    ]
    
    return result

class Long:
    def __init__(self, long):
        if isinstance(long, Long):
            self.data = long.data
        else:
            self.data = long
        self.indices = {}
        self.amortize = True
    @classmethod
    def from_data_array(cls, data_array):
        return cls(flatten_data_array_coords_and_values(data_array))
        data = [
            tuple._asdict()
            for tuple in data_array.to_dataframe().reset_index().itertuples(index=False)
        ]
        return cls(data)
    
    @classmethod
    def from_data_frame(cls, data_frame):
        return cls(data_frame.to_dict(orient='records'))
    def remove_cols(self, *cols):
        return Long([
            {k:v for k,v in record.items() if k not in cols}
            for record in self.data
        ])
    def rename(self, **name_map):
        '''
        Args:
            name_map - Dict[str, str] - mapping of old names to new names
        '''
        return Long([
            {name_map.get(k, k): v for k,v in record.items()}
            for record in self.data
        ])
    def add_index(self, keys):
        '''
        Adds keys as an index.
        Args:
            keys - List[str] - list of field names to be an index
        '''
        keys = tuple(keys)
        self.indices[keys] = self.build_index(keys)
    
    def build_index(self, keys):
        index = defaultdict(list)
        for i, record in enumerate(self.data):
            index[tuple(record[k] for k in keys)].append(i)
        return index

    def factorize(self, keys, col_field, value_field):
        '''
        Essentially a wide-to-long operation.
        If I have three metrics A, B, C per row, I can use this to factorize them into a long format.
        So that I get three rows per original row, one for each metric.

        Args:
            keys - List[str] - list of field names to factorize
            col_field - str - name of the field to store the keys
            value_field - str - name of the field to store the values
        '''
        new_data = []
        for key in keys:
            new_data.extend([{
                **{
                    k:v 
                    for k,v in record.items()
                    if k not in keys
                },
                col_field: key,
                value_field: record[key]
            } for record in self.data])
        return Long(new_data)
    def unfactorize(self, col_field, value_field, group_keys=None):
        import pandas as pd
        all_cols = list(self.data[0].keys())
        assert col_field in all_cols and value_field in all_cols
        if group_keys is None:
            group_keys = [k for k in all_cols if k not in [col_field, value_field]]
        return Long.from_data_frame(pd.DataFrame(self.data).pivot(index=group_keys, columns=col_field, values=value_field).reset_index())
            
    @classmethod
    def concat(cls, longs):
        long_data = [
            long.data if isinstance(long, Long) else long
            for long in longs
        ]
        return cls([
            record
            for long in long_data
            for record in long
        ])
    def insert_col(self, field, values):
        assert len(values) == len(self.data)
        return Long([
            {**record, field: value}
            for record, value in zip(self.data, values)
        ])
    def attach(self, **kwargs):
        '''
        Attach additional fields to each record.
        Fields will silently overwrite existing fields.
        Returns shallow copies of records
        '''
        return Long([
            {**record, **kwargs}
            for record in self.data
        ])

    def get(self, **keys):
        r = self.get_all(**keys)
        assert len(r) == 1
        return r[0]
    def unique(self, key):
        if (key,) in self.indices:
            return sorted([k for k, in self.indices[(key,)].keys()])
        r = set(record[key] for record in self.data)
        return sorted(list(r))
    def _find_or_build_index_keys(self, keys):
        keys = tuple(keys)
        for index_keys in self.indices:
            if set(index_keys) == set(keys):
                return index_keys
        if self.amortize:
            self.add_index(keys)
            return tuple(keys)
        return None
    def merge_cols(self, col_name, cols):
        col_type = namedtuple(col_name, cols)
        return Long([
            {
                **{k: v for k, v in record.items() if k not in cols},
                col_name: col_type(*[record[k] for k in cols])
            }
            for record in self.data
        ])
    def unmerge_cols(self, col_name, cols=None):
        return Long([
            {
                **{k:v for k, v in record.items() if k != col_name},
                **(
                    record[col_name]._asdict()
                    if cols is None 
                    else {
                        k: v
                        for k, v in zip(cols, record[col_name], strict=True)
                    }
                )
            }
            for record in self.data
        ])
    def get_all(self, **keys):
        index_keys = self._find_or_build_index_keys(keys.keys())
        if index_keys is not None:
            r = [
                self.data[i]
                for i in self.indices[index_keys][tuple(keys.values())]
            ]
        else:
            r = [
                record
                for record in self.data
                if all(record[k] == v for k,v in keys.items())
            ]
        return Long(r)
    def to_data_array(self, coords, value=None):
        '''
        Choose a list of coords to be the dimensions of the data array,
        and a value to be the data.

        Args:
            coords - List[str] - list of field names to be dimensions
            value - str - field name to be the data

        Returns:
            xr.DataArray - xarray data
        '''
        import xarray as xr
        import numpy as np
        import pandas as pd
        coord_names = coords
        coord_values = [
            (name, self.unique(name))
            for name in coord_names
        ]
        coord_maps = {
            name: {v:i for i,v in enumerate(values)}
            for name, values in coord_values
        }
        data = np.empty(
            [len(v) for n,v in coord_values],
            dtype=object
        )
        for record in self.data:
            index = tuple(
                coord_maps[name][record[name]]
                for name in coord_names
            )
            data[index] = record[value] if value is not None else record
        # convert tuple to multiindex
        def convert_to_multiindex(values):
            # return values
            if len(values) == 0:
                return values
            if isinstance(values[0], tuple):
                return pd.MultiIndex.from_tuples(values, names=values[0]._fields)
            else:
                return values
        coord_values = [
            (name, convert_to_multiindex(values))
            for name, values in coord_values
        ]
        return xr.DataArray(
            data.tolist(),
            coords=coord_values,
            name=value
        )
    def col(self, field):
        return [
            record[field]
            for record in self.data
        ]
    def clear_index(self):
        self.indices = {}
    def __getitem__(self, row_index):
        return self.data[row_index]
    def __len__(self):
        return len(self.data)
    def __setitem__(self, row_index, value):
        self.data[row_index] = value
        self.clear_index()
    def __iter__(self):
        return iter(self.data)

# class Atom:
#     def __init__(self, obj):
#         self._obj = obj
#     def __repr__(self):
#         return f'Atom<{repr(self._obj)}>'
# class DictIndex:
#     def __init__(self, label : str, values : list[str]):
#         self.label = label
#         self.values = values
# class ListIndex:
#     def __init__(self, label : str):
#         self.label = label
# class Tensorial:
#     def __init__(self, struct, labels):
#         self.array, self.indices = traverse(struct, labels) 
#         # constructs np.ndarray of Atom.
#         # and also a list of Union[DictIndex, ListIndex]
#     def rearrange(self, new_labels):
#         # permute labels
#         pass
#     def relabel(self, new_labels):
#         # relabel labels
#         pass
#     def flatten(self, list_label, *dict_labels):
#         # stuff dict_labels into list_label
#     def unflatten(self, list_label, *dict_labels):
#         # explode dict labels out.
#         # will replace list_label -> *dict_labels, list_label 
#     def dump(self):
#         # get rid of atom, try to get a numpy array