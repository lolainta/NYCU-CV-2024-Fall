import os
import cv2

data_dirname = 'data_5'
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(dir_path, data_dirname)

def create_label(image_path):
    # Mouse callback function to get coordinates
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            coords.append((x, y))
            cv2.destroyAllWindows()

    # Read and resize the image
    image = cv2.imread(image_path)
    original_size = (image.shape[1], image.shape[0])
    resized_image = cv2.resize(image, (1280, 720))
    coords = []
    cv2.imshow('Image', resized_image)
    cv2.setMouseCallback('Image', click_event)
    cv2.waitKey(0)
    
    # Get label from coordinates
    if coords:
        # Adjust coordinates back to original size
        x, y = coords[0]
        x = int(x * original_size[0] / 1280)
        y = int(y * original_size[1] / 720)
        label = f"{x},{y}"
    else:
        label = "No coordinates clicked"
    
    return label

def get_label():
    labels = {}
    for filename in os.listdir(data_path):
        if filename.endswith(('.jpg')):
            image_path = os.path.join(data_path, filename)
            label = create_label(image_path)
            labels[filename] = label
            
    # Save labels to a file
    with open(os.path.join(data_path, 'labels.txt'), 'w') as f:
        for filename, label in labels.items():
            f.write(f"{filename}: {label}\n")   
        
    return labels

def parse_labels():
    labels_file_path = os.path.join(data_path, 'labels.txt')
    labels = {}
    with open(labels_file_path, 'r') as f:
        for line in f:
            filename, label = line.strip().split(': ')
            labels[filename] = label
    return labels

def test_label(labels):
    for filename, label in labels.items():
        if label != "No coordinates clicked":
            x, y = map(int, label.split(','))
            image_path = os.path.join(data_path, filename)
            image = cv2.imread(image_path)
            cv2.circle(image, (x, y), 10, (255, 0, 0), 20)
            image = cv2.resize(image, (1280, 720))
            cv2.imshow('Labeled Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def main():
    if os.path.exists(os.path.join(dir_path, data_dirname, 'labels.txt')):
        labels = parse_labels()
    else:
        labels = get_label()
    test_label(labels)

if __name__ == "__main__":
    main()