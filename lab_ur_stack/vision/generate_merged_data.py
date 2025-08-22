import os
import json
import shutil

# Define the root directory where your "collected_images" directory is located
root_dir = '../scripts/collected_images'
merged_data_dir = 'images_data_merged_hires'

# Ensure the merged_data directory exists
if not os.path.exists(merged_data_dir):
    os.makedirs(merged_data_dir)

# Create subdirectories for images and depth images
images_dir = os.path.join(merged_data_dir, 'images')
depth_dir = os.path.join(merged_data_dir, 'depth')

if not os.path.exists(images_dir):
    os.makedirs(images_dir)
if not os.path.exists(depth_dir):
    os.makedirs(depth_dir)

merged_metadata = []

image_counter = 1
depth_counter = 1

# Iterate through all directories in the root directory
for experiment_dir in os.listdir(root_dir):
    if experiment_dir.startswith("5blocks_hires"):
        experiment_path = os.path.join(root_dir, experiment_dir)
        metadata_file = os.path.join(experiment_path, 'metadata.json')

        # Load the metadata
        with open(metadata_file, 'r') as file:
            metadata = json.load(file)

        block_positions = metadata["block_positions"]

        # Move and rename image and depth files and create metadata entries
        for i, (image_file, depth_file) in enumerate(zip(metadata["images_rgb"], metadata["images_depth"])):
            src_image_path = os.path.join(experiment_path, image_file)
            dst_image_path = os.path.join(images_dir, f'image_{image_counter}.npy')
            shutil.copyfile(src_image_path, dst_image_path)

            src_depth_path = os.path.join(experiment_path, depth_file)
            dst_depth_path = os.path.join(depth_dir, f'depth_{depth_counter}.npy')
            shutil.copyfile(src_depth_path, dst_depth_path)

            merged_metadata.append({
                "image_rgb": f'image_{image_counter}.npy',
                "image_depth": f'depth_{depth_counter}.npy',
                "ur5e_1_config": metadata["ur5e_1_config"][i],
                "ur5e_2_config": metadata["ur5e_2_config"][i],
                "block_positions": block_positions
            })

            image_counter += 1
            depth_counter += 1

# Save the merged metadata to a JSON file
with open(os.path.join(merged_data_dir, 'merged_metadata.json'), 'w') as file:
    json.dump(merged_metadata, file, indent=4)

print("Data merging complete.")
