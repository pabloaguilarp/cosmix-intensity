import pandas as pd
import numpy as np

def save_pcd_with_pandas(filename,
                         coords,
                         intensity,
                         labels):
    num_points = coords.shape[0]

    # Create a Pandas DataFrame
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
        "intensity": intensity,
        "labels": labels
    })

    # Define the PCD header
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity labels
SIZE 4 4 4 4 4
TYPE F F F F F
COUNT 1 1 1 1 1
WIDTH {num_points}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {num_points}
DATA ascii
"""

    # Write the header and the data to the file
    with open(filename, "w") as f:
        f.write(header)
        df.to_csv(f, sep=" ", index=False, header=False, float_format="%.6f")

def save_pcd_binary(filename, coords, intensity, labels):
    num_points = coords.shape[0]

    # Define the binary PCD header
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity labels
SIZE 4 4 4 4 4
TYPE F F F F F
COUNT 1 1 1 1 1
WIDTH {num_points}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {num_points}
DATA binary
"""

    # Combine XYZ, intensity, and labels into a structured NumPy array
    dtype = np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('intensity', np.float32), ('labels', np.float32)
    ])
    data = np.zeros(num_points, dtype=dtype)
    data['x'], data['y'], data['z'] = coords[:, 0], coords[:, 1], coords[:, 2]
    data['intensity'] = intensity.squeeze().astype(np.float32)
    data['labels'] = labels.squeeze().astype(np.float32)

    # Write header and binary data to the file
    with open(filename, "wb") as f:
        f.write(header.encode("utf-8"))  # Write the header as ASCII
        f.write(data.tobytes())  # Write the binary data