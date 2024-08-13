#!/usr/bin/env python
# coding: utf-8

# In[13]:


import cv2
import numpy as np
import os

def match_distance(matches):
    distances = [match.distance for match in matches]
    threshold = np.median(distances)
    filtered_matches = [match for match in matches if match.distance < threshold]
    return filtered_matches

def filter_matches_by_distance(matches, threshold):
    good_matches = []
    distances = [match.distance for match in matches]
    if distances:
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        threshold_distance = mean_distance + threshold * std_distance
        for match in matches:
            if match.distance < threshold_distance:
                good_matches.append(match)
    return good_matches

def process_image_pair(image1_path, image2_path, similarity_threshold=0.2):
    # Read two images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Create SIFT objects
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # Convert descriptors to uint8
    descriptors1 = (descriptors1 * 255).astype(np.uint8)
    descriptors2 = (descriptors2 * 255).astype(np.uint8)

    # Create a Brute-Force Matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Match descriptors using BFMatcher
    matches = bf.match(descriptors1, descriptors2)

    # Filter matches based on distance using the new function
    filtered_matches = filter_matches_by_distance(matches, threshold=30)

    # Filter matches based on distance using the original function
    matches = match_distance(matches)

    # RANSAC
    if len(filtered_matches) >= 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in filtered_matches]).reshape(-1, 1, 2)
        
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 50)
        mask = mask.ravel()
        good_matches = [filtered_matches[i] for i in range(len(mask)) if mask[i]]

        similarity_score = len(good_matches) / min(len(keypoints1), len(keypoints2))

        # Print or use the similarity score as needed
        print("Similarity Score for", image1_path, "and", image2_path, ":", similarity_score)

        # Check if the similarity score is above the threshold
        if similarity_score > similarity_threshold:
            print("The pair is considered similar.")
        else:
            print("The pair is not considered similar.")

        # Draw matches with RANSAC and Distance Thresholding
        img_matches_with_ransac = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None,
                                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Display the matched images with RANSAC only
        cv2.imshow("Matches With RANSAC and Distance Thresholding", img_matches_with_ransac)
        cv2.waitKey(20)
        cv2.destroyAllWindows()
    else:
        print("Not enough matches for RANSAC in", image1_path, "and", image2_path)

# Specify the folder containing the images
folder_path = 'C:/Graduation_project/computer_vision/lab6assignment/assignmentdata/'

# Get a list of all image files in the folder
image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
# Iterate through all adjacent pairs of images
for i in range(len(image_files) - 1):
    image1_path = os.path.join(folder_path, image_files[i])
    image2_path = os.path.join(folder_path, image_files[i + 1])
    process_image_pair(image1_path, image2_path)


# In[ ]:




