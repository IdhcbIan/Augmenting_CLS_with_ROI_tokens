import os
import pickle

#assuming modal volume put cub-data ./roxford5k_converted /roxford5k_converted



# Load the ground truth data from pickle file
with open("roxford5k/gnd_roxford5k.pkl", "rb") as f:
    data = pickle.load(f)

imlist = data['imlist']  # Database images
qimlist = data['qimlist']  # Query images
gnd = data['gnd']

print(f"Database images: {len(imlist)}")
print(f"Query images: {len(qimlist)}")
print(f"Ground truth entries: {len(gnd)}")

# Extract unique building names from all images
all_images = imlist + qimlist
building_names = set()
for img_name in all_images:
    building = '_'.join(img_name.split('_')[:-1])  # Everything except the last number part
    building_names.add(building)

building_names = sorted(list(building_names))
print(f"Found {len(building_names)} unique buildings/locations: {building_names}")

# Create building name to class ID mapping
building_to_class = {building: i for i, building in enumerate(building_names)}

# Create directory structure
os.system("mkdir -p roxford5k_converted")

# Create class directories
for i in range(len(building_names)):
    os.system(f"mkdir -p roxford5k_converted/{i}")

print(f"Created {len(building_names)} class directories")

# Copy all images to their respective class directories
print("Copying images to class directories...")
all_image_info = []

# Process database images
for img_idx, img_name in enumerate(imlist):
    building = '_'.join(img_name.split('_')[:-1])
    class_id = building_to_class[building]
    
    src_path = f"roxford5k/jpg/{img_name}.jpg"
    dst_path = f"roxford5k_converted/{class_id}/{img_name}.jpg"
    os.system(f"cp {src_path} {dst_path}")
    
    all_image_info.append({
        'global_idx': img_idx,
        'img_name': img_name,
        'class_id': class_id,
        'building': building,
        'path': dst_path,
        'type': 'database'
    })

# Process query images (continue global indexing)
for q_idx, qimg_name in enumerate(qimlist):
    building = '_'.join(qimg_name.split('_')[:-1])
    class_id = building_to_class[building]
    
    src_path = f"roxford5k/jpg/{qimg_name}.jpg"
    dst_path = f"roxford5k_converted/{class_id}/{qimg_name}.jpg"
    os.system(f"cp {src_path} {dst_path}")
    
    all_image_info.append({
        'global_idx': len(imlist) + q_idx,
        'img_name': qimg_name,
        'class_id': class_id,
        'building': building,
        'path': dst_path,
        'type': 'query'
    })

print("Finished copying images.")

# Create classes.txt (class_id to class_id mapping, same format as Flowers)
with open("roxford5k_converted/classes.txt", "w") as f:
    for i, building_name in enumerate(building_names):
        print(f"{i} {i}", file=f)

# Create building_names.txt (class_id to building_name mapping for reference)
with open("roxford5k_converted/building_names.txt", "w") as f:
    for i, building_name in enumerate(building_names):
        print(f"{i} {building_name}", file=f)

# Create image_class_labels.txt (image_id to class_id mapping)
with open("roxford5k_converted/image_class_labels.txt", "w") as f:
    for img_info in all_image_info:
        print(f"{img_info['global_idx']} {img_info['class_id']}", file=f)

# Create images.txt (image_id to path mapping)
with open("roxford5k_converted/images.txt", "w") as f:
    for img_info in all_image_info:
        print(f"{img_info['global_idx']} {img_info['path']}", file=f)

# Create query_images.txt (query image information)
with open("roxford5k_converted/query_images.txt", "w") as f:
    for img_info in all_image_info:
        if img_info['type'] == 'query':
            print(f"{img_info['global_idx']} {img_info['path']} {img_info['img_name']}", file=f)

# Create database_images.txt (database image information) 
with open("roxford5k_converted/database_images.txt", "w") as f:
    for img_info in all_image_info:
        if img_info['type'] == 'database':
            print(f"{img_info['global_idx']} {img_info['path']} {img_info['img_name']}", file=f)

# Create ground_truth.txt (ground truth information for queries)
with open("roxford5k_converted/ground_truth.txt", "w") as f:
    for i, gnd_entry in enumerate(gnd):
        bbx = gnd_entry['bbx']
        easy = gnd_entry['easy']
        hard = gnd_entry['hard']
        junk = gnd_entry['junk']
        
        # Format: query_id bbx_x1,bbx_y1,bbx_x2,bbx_y2 easy_indices hard_indices junk_indices
        bbx_str = f"{bbx[0]},{bbx[1]},{bbx[2]},{bbx[3]}"
        easy_str = ",".join(map(str, easy)) if easy else ""
        hard_str = ",".join(map(str, hard)) if hard else ""
        junk_str = ",".join(map(str, junk)) if junk else ""
        
        print(f"{i} {bbx_str} {easy_str} {hard_str} {junk_str}", file=f)

# Create train_test_split.txt (same format as Flowers: 1=train, 0=test)
# For retrieval evaluation, we'll mark most as train and some as test
with open("roxford5k_converted/train_test_split.txt", "w") as f:
    for img_info in all_image_info:
        # Use same pattern as Flowers: most images are train (1), last few in each class are test (0)
        building = img_info['building']
        class_id = img_info['class_id']
        
        # Count images in this class to determine train/test split
        class_images = [x for x in all_image_info if x['class_id'] == class_id]
        img_position_in_class = [x['global_idx'] for x in class_images].index(img_info['global_idx'])
        
        # Last 20% of images in each class are test, rest are train (similar to Flowers pattern)
        if img_position_in_class >= len(class_images) * 0.8:
            split_flag = 0  # test
        else:
            split_flag = 1  # train
            
        print(f"{img_info['global_idx']} {split_flag}", file=f)

print("Conversion completed!")
print("Files created:")
print("- roxford5k_converted/classes.txt (same format as Flowers)")
print("- roxford5k_converted/building_names.txt (class to building mapping)")
print("- roxford5k_converted/image_class_labels.txt (image to class mapping)")
print("- roxford5k_converted/images.txt (image to path mapping)")
print("- roxford5k_converted/query_images.txt (query images)")
print("- roxford5k_converted/database_images.txt (database images)")
print("- roxford5k_converted/ground_truth.txt (retrieval ground truth)")
print("- roxford5k_converted/train_test_split.txt (train/test split)")
print(f"- roxford5k_converted/0-{len(building_names)-1}/ (class directories)")

# Print class distribution
print("\nClass distribution:")
class_counts = {}
for img_info in all_image_info:
    class_id = img_info['class_id']
    building = img_info['building']
    if class_id not in class_counts:
        class_counts[class_id] = {'building': building, 'count': 0}
    class_counts[class_id]['count'] += 1

for class_id in sorted(class_counts.keys()):
    building = class_counts[class_id]['building']
    count = class_counts[class_id]['count']
    print(f"Class {class_id} ({building}): {count} images")
