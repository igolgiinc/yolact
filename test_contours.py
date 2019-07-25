# Import required packages:
import cv2

# Load the image and convert it to grayscale:
image = cv2.imread("./test_images/contour_test.png")
print("image shape:", image.shape)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("gray_image shape:", gray_image.shape)
# Apply cv2.threshold() to get a binary image
ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
print("thresh shape:", thresh.shape)

'''
for i in range(thresh.shape[0]):
    for j in range(thresh.shape[1]):
        print("%d " % (thresh[i,j]), end =" ")

print()
'''

# Find contours:
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Draw contours:
cv2.drawContours(image, contours, 0, (0, 255, 0), 2)

# Calculate image moments of the detected contour
M = cv2.moments(contours[0])

# Print center (debugging):
print("center X : '{}'".format(round(M['m10'] / M['m00'])))
print("center Y : '{}'".format(round(M['m01'] / M['m00'])))

# Draw a circle based centered at centroid coordinates
cv2.circle(image, (round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])), 5, (0, 255, 0), -1)

cv2.imwrite("/mnt/bigdrive1/cnn/outputs/contour_test_output.png", image)

# Show image:
#cv2.imshow("outline contour & centroid", image)

# Wait until a key is pressed:
#cv2.waitKey(0)

# Destroy all created windows:
#cv2.destroyAllWindows()
