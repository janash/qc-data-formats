import h5py
import numpy as np
import json
from collections import Counter
import tqdm
import sqlite3
from qcportal import load_dataset_view

import os

def process_dataset(dsv, dataset_name, spec_counts):
    """Process the entire dataset and save to a single HDF5 file."""
    
    output_file = f"{dataset_name}.h5"
    
    with h5py.File(output_file, "w") as hdf:
        # Add dataset metadata
        hdf.attrs["name"] = dsv.name
        hdf.attrs["description"] = dsv.description
        hdf.attrs["version"] = "0.1"
        hdf.attrs["metadata"] = json.dumps(dsv.metadata)
        
        # Create group for entries/molecules
        entries = hdf.create_group("entries")
        
        # Process the entries
        print("Processing entries...")
        element_counter = Counter()
        entry_map = {}
        
        for i, entry in tqdm.tqdm(enumerate(dsv.iterate_entries())):
            # create a group for this entry
            entry_id = f"entry_{i}"
            entry_group = entries.create_group(entry_id)
            
            # Store mapping between entry name and ID
            entry_map[entry.name] = entry_id

            # add entry name
            entry_group.attrs["entry_name"] = entry.name

            # add molecule identifiers as attributes
            qcel_molecule = entry.molecule.identifiers.dict()
            for key, value in qcel_molecule.items():
                if value:
                    entry_group.attrs[key] = value

            # add charge, multiplicity, and other properties
            entry_group.attrs["molecular_charge"] = entry.molecule.molecular_charge
            entry_group.attrs["molecular_multiplicity"] = entry.molecule.molecular_multiplicity
            entry_group.attrs["element_composition"] = json.dumps(entry.molecule.element_composition()) 
            
            # Update element counter for dataset-wide statistics
            current_elements = Counter(entry.molecule.element_composition())
            element_counter.update(current_elements)

            # add elements, connectivity, and geometry information
            entry_group.create_dataset("symbols", data=np.array(entry.molecule.symbols, dtype='S'))
            entry_group.create_dataset("atomic_numbers", data=np.array(entry.molecule.atomic_numbers, dtype=np.int32))
            entry_group.create_dataset("geometry", data=np.array(entry.molecule.geometry, dtype=np.float64))
            if entry.molecule.connectivity is not None:
                connectivity_array = np.array(entry.molecule.connectivity, dtype=np.int32)
                entry_group.create_dataset("connectivity", data=connectivity_array)
            else:
                entry_group.create_dataset("connectivity", data=np.empty((0, 3), dtype=np.int32))
            
            # Add fragments as a dataset
            fragments_array = np.array(entry.molecule.fragments, dtype=np.int32)
            entry_group.create_dataset("fragments", data=fragments_array)
        
        inverse_map = {v: k for k, v in entry_map.items()}
        entries.attrs["entries_map"] = json.dumps(inverse_map)
        # Add elemental composition at the top level
        hdf.attrs["element_composition"] = json.dumps(dict(element_counter))
        
        # Create specifications group
        specs = hdf.create_group("specifications")
        spec_map = {}
        
        # Process each specification
        for i, spec_name in enumerate(dsv.specification_names):
            print(f"Processing specification: {spec_name}")
            spec_id = f"spec_{i}"
            process_specification(dsv, hdf, specs, spec_name, spec_id, entry_map, spec_counts)

            spec_id = f"specification_{i}"
            spec_map[spec_name] = spec_id
        
        inverse_spec_map = {v: k for k, v in spec_map.items()}
        specs.attrs["specifications_map"] = json.dumps(spec_map)
        print(f"Dataset saved to {output_file}")
        return output_file

def process_specification(dsv, hdf, specs_group, spec_name, spec_id, entry_map, spec_counts):
    """Process a single specification and add it to the HDF5 file."""
    
    # Create the specification group
    spec_group = specs_group.create_group(spec_id)
    spec_group.attrs["specification_name"] = spec_name
    
    # Add specification metadata
    spec_group.attrs["driver"] = dsv.specifications[spec_name].specification.driver.name
    spec_group.attrs["program"] = dsv.specifications[spec_name].specification.program
    spec_group.attrs["method"] = dsv.specifications[spec_name].specification.method
    spec_group.attrs["basis"] = dsv.specifications[spec_name].specification.basis or ""
    
    # Create properties group for aggregated properties
    props_group = spec_group.create_group("properties")
    
    # First pass analysis to determine property types
    print("Analyzing records to determine property types...")
    record_samples = 0
    max_samples = 100

    properties_map = {}
    compound_properties = []
    none_types = []
    array_properties = []
    fixed_shape_arrays = {}
    variable_arrays = []

    for record in dsv.iterate_records(specification_names=spec_name):
        if record_samples >= max_samples:
            break
            
        properties = record[2].properties
        record_samples += 1
        
        for key, value in properties.items():
            if key not in properties_map:
                # First encounter of this property
                if value is None:
                    properties_map[key] = {"dtype": None}
                    none_types.append(key)
                elif isinstance(value, dict):
                    properties_map[key] = {"dtype": "compound"}
                    for key2, value2 in value.items():
                        arr = np.array(value2)
                        properties_map[key][key2] = {"dtype": arr.dtype}
                    compound_properties.append(key)
                elif isinstance(value, str):
                    try:
                        float_value = float(value)
                        
                        # Check if integer
                        if float_value.is_integer():
                            properties_map[key] = {"dtype": np.dtype(np.int32)}
                        else:
                            properties_map[key] = {"dtype": np.dtype(np.float64)}
                    except ValueError:
                        properties_map[key] = {"dtype": np.dtype('S')}
                else:
                    # Handle potential array or scalar
                    arr = np.array(value)
                    properties_map[key] = {"dtype": arr.dtype}
                    if arr.shape:  # It's an array
                        if key not in array_properties:
                            array_properties.append(key)
                            fixed_shape_arrays[key] = arr.shape
                        
            elif key in array_properties and key not in variable_arrays:
                # Check if this array has consistent shape with previous samples
                arr = np.array(value)
                if arr.shape and arr.shape != fixed_shape_arrays.get(key):
                    # Found inconsistent shape, mark as variable
                    variable_arrays.append(key)
                    # Remove from fixed_shape_arrays
                    if key in fixed_shape_arrays:
                        del fixed_shape_arrays[key]

    # Determine record count for pre-allocation
    record_count = spec_counts.get(spec_name, 0)
    
    # Create the property datasets
    print("Creating datasets for properties based on analysis...")
    for key, value in properties_map.items():
        if value["dtype"] is None:
            # Skip None types
            pass
        elif value["dtype"] == "compound":
            # Handle compound properties with vlen
            compound_keys = [k for k in value.keys() if k != "dtype"]
            for k in compound_keys:
                base_dtype = value[k]["dtype"]
                vlen_type = h5py.special_dtype(vlen=base_dtype)
                props_group.create_dataset(f"{key}_{k}", shape=(record_count,), dtype=vlen_type, 
                                        chunks=True, compression="gzip", compression_opts=4)
        elif key in variable_arrays:
            # Handle variable-shape arrays with vlen
            base_dtype = value["dtype"]
            vlen_type = h5py.special_dtype(vlen=base_dtype)
            props_group.create_dataset(key, shape=(record_count,), dtype=vlen_type, 
                                    chunks=True, compression="gzip", compression_opts=4)
        elif key in fixed_shape_arrays:
            # Handle fixed-shape arrays with known dimensions
            shape = fixed_shape_arrays[key]
            full_shape = (record_count,) + shape
            props_group.create_dataset(key, shape=full_shape, dtype=value["dtype"], 
                                    chunks=True, compression="gzip", compression_opts=4)
        else:
            # Handle scalar properties
            props_group.create_dataset(key, shape=(record_count,), dtype=value["dtype"], 
                                    chunks=True, compression="gzip", compression_opts=4)
    
    # Create dataset for entry references
    dt = h5py.string_dtype(encoding='utf-8')
    entry_refs_dset = spec_group.create_dataset("entry_references", shape=(record_count,), dtype=dt, 
                                            chunks=True, compression="gzip", compression_opts=4)
    
    # Fill the data
    fill_spec_data(dsv, spec_name, props_group, entry_refs_dset, entry_map, 
                  compound_properties=compound_properties, variable_arrays=variable_arrays)
    
    # Add metadata
    spec_group.attrs['record_count'] = record_count
    spec_group.attrs['none_types'] = json.dumps(none_types)
    spec_group.attrs['compound_properties'] = json.dumps(compound_properties)
    spec_group.attrs['variable_arrays'] = json.dumps(variable_arrays)
    spec_group.attrs['fixed_shape_arrays'] = json.dumps({k: list(v) for k, v in fixed_shape_arrays.items()})

def fill_spec_data(dsv, spec_name, props_group, entry_refs_dset, entry_map, compound_properties=None, variable_arrays=None):
    """Fill specification data with actual values"""
    
    compound_properties = compound_properties or []
    variable_arrays = variable_arrays or []
    
    # Batch size
    batch_size = 5000
    
    # Get total record count
    record_count = entry_refs_dset.shape[0]
    
    # Prepare data containers for batches
    batch_entry_refs = []
    batch_data = {}
    
    # Initialize data containers for each property
    for key in props_group.keys():
        if key.split('_')[0] not in compound_properties:
            batch_data[key] = []
    
    # For compound properties
    for key in compound_properties:
        compound_keys = [k.split('_')[1] for k in props_group.keys() 
                        if k.startswith(f"{key}_")]
        for k in compound_keys:
            batch_data[f"{key}_{k}"] = []
    
    batch_count = 0
    processed_count = 0
    
    # Process in batches
    print(f"Processing in batches of {batch_size} records...")
    for record_idx, record in tqdm.tqdm(enumerate(dsv.iterate_records(specification_names=spec_name))):
        properties = record[2].properties
        entry_name = record[0]
        
        # Add entry reference to batch
        batch_entry_refs.append(entry_map[entry_name])
        
        # Add property data to batch
        for key, value in properties.items():
            if value is None:
                # Skip None values
                continue
            elif key in compound_properties:
                # Handle compound properties
                for sub_key, sub_value in value.items():
                    dataset_name = f"{key}_{sub_key}"
                    if dataset_name in batch_data:
                        batch_data[dataset_name].append(sub_value)
            elif key in props_group:
                # Handle all other properties
                batch_data[key].append(value)
                
        batch_count += 1
        
        # Write batch when it reaches batch_size or at the end
        if batch_count >= batch_size or record_idx == record_count - 1:
            start_idx = processed_count
            end_idx = processed_count + batch_count
            
            # Write entry references
            entry_refs_dset[start_idx:end_idx] = batch_entry_refs
            
            # Write property data
            for key, values in batch_data.items():
                if values and key in props_group:
                    # Different handling based on property type
                    if key.split('_')[0] in compound_properties or key in variable_arrays:
                        # Write variable-length arrays one by one (still faster in batches)
                        for i, val in enumerate(values):
                            props_group[key][start_idx + i] = val
                    else:
                        # Try to write fixed-shape arrays in one operation
                        try:
                            props_group[key][start_idx:end_idx] = values
                        except (ValueError, TypeError):
                            # Fall back to one-by-one if shapes don't match
                            for i, val in enumerate(values):
                                props_group[key][start_idx + i] = val
            
            # Update processed count and reset batch
            processed_count += batch_count
            batch_count = 0
            batch_entry_refs = []
            for key in batch_data:
                batch_data[key] = []
                
            print(f"Processed {processed_count}/{record_count} records")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Process a QCArchive dataset into a single HDF5 file.")
    parser.add_argument("-sqlite_file", type=str,
                        help="Location of the SQLite file to load the dataset from")
    
    args = parser.parse_args()
    sqlite_file = args.sqlite_file

    dataset_name = os.path.basename(args.sqlite_file).replace(".sqlite", "")

    conn = sqlite3.connect(sqlite_file)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            specification_name, 
            COUNT(*) AS record_count 
        FROM 
            dataset_records 
        GROUP BY 
            specification_name 
        ORDER BY 
            record_count DESC
    """)
    spec_counts = cursor.fetchall()
    conn.close()
    spec_counts = {k[0]: k[1] for k in spec_counts}
    
    dsv = load_dataset_view(sqlite_file)
    
    # Process the entire dataset into a single file
    output_file = process_dataset(dsv, dataset_name, spec_counts)
    
    print(f"Successfully created {output_file}")