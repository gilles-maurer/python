from PIL import Image

single_threshold = 128
double_threshold_min = 64
double_threshold_max = 172

def to_gray_single_thresholding(image):
    width, height = image.size
    new_image = Image.new("1", (width, height))  # mode : black and white, 1-bit pixels

    for i in range(width):
        for j in range(height):
            pixel = image.getpixel((i, j))
            new_pixel_value = 0 if (pixel[0] + pixel[1] + pixel[2]) / 3 < single_threshold else 1
            new_image.putpixel((i, j), new_pixel_value)

    return new_image


def to_gray_double_thresholding(image):
    width, height = image.size
    new_image = Image.new("1", (width, height))  # mode : black and white, 1-bit pixels

    for i in range(width):
        for j in range(height):
            pixel = image.getpixel((i, j))
            temp = (pixel[0] + pixel[1] + pixel[2]) / 3
            if temp < double_threshold_min or temp > double_threshold_max:
                new_image.putpixel((i, j), 0)
            else:
                new_image.putpixel((i, j), 1)

    return new_image


def to_greyscale(image):
    width, height = image.size
    new_image = Image.new("L", (width, height))  # mode : black and white, 8-bit pixels

    for i in range(width):
        for j in range(height):
            pixel = image.getpixel((i, j))
            new_pixel_value = int(0.2989 * pixel[0] + 0.5870 * pixel[1] + 0.1140 * pixel[2])
                            # better than "val = int((px[0] + px[1] + px[2]) / 3)"
            new_image.putpixel((i, j), new_pixel_value)

    return new_image


def histogramEqualization(image):
    width, height = image.size
    new_image = Image.new("L", (width, height))  # mode : black and white, 8-bit pixels

    # Init the range of each pixel
    counts = {}
    for i in range(256):
        counts[i] = 0

    for i in range(width):
        for j in range(height):
            counts[image.getpixel((i, j))] += 1

    # cumulative sum
    cumulSum = []
    cumulSum.append(counts[0])
    for i in range(1, 256):
        cumulSum.append(cumulSum[-1] + counts[i])

    # Normalize cumulative sum
    min = counts[0]
    max = cumulSum[255]

    for i in range(256):
        cumulSum[i] = int(((cumulSum[i] - min) * 255) / (max - min))

    # Put new values in the image
    for i in range(width):
        for j in range(height):
            new_image.putpixel((i, j), cumulSum[image.getpixel((i, j))])

    return new_image


def meanFilter(image):
    mask_size = 71
    half_mask_size = int((mask_size - 1) / 2)

    width, height = image.size
    new_image = Image.new("L", (width, height))  # mode : black and white, 8-bit pixels

    print("Pre-process summed area table")

    summed_area_table = [[0.0] * height for _ in range(width)]  # initialization of the table

    for i in range(width):
        for j in range(height):
            summed_area_table[i][j] = image.getpixel((i, j))
            if i > 0:
                summed_area_table[i][j] += summed_area_table[i - 1][j]
            if j > 0:
                summed_area_table[i][j] += summed_area_table[i][j - 1]
            if i > 0 and j > 0:
                summed_area_table[i][j] -= summed_area_table[i - 1][j - 1]

    print("Build new image")

    # build new image
    for i in range(width):
        for j in range(height):
            a = [(i - half_mask_size) if (i - half_mask_size) > 0 else 0,
                 (j - half_mask_size) if (j - half_mask_size) > 0 else 0]
            b = [(i + half_mask_size) if (i + half_mask_size) < width else width-1,
                 (j - half_mask_size) if (j - half_mask_size) > 0 else 0]
            c = [(i - half_mask_size) if (i - half_mask_size) > 0 else 0,
                 (j + half_mask_size) if (j + half_mask_size) < height else height-1]
            d = [(i + half_mask_size) if (i + half_mask_size) < width else width-1,
                 (j + half_mask_size) if (j + half_mask_size) < height else height-1]

            new_pixel_value = (summed_area_table[d[0]][d[1]] - summed_area_table[b[0]][b[1]] -
                               summed_area_table[c[0]][c[1]] + summed_area_table[a[0]][a[1]]) / (mask_size*mask_size)

            new_image.putpixel((i, j), int(new_pixel_value))


    return new_image


images = ["./yoda.jpeg", "./road.jpg"]

with Image.open(images[0]) as yoda_image:
   to_gray_single_thresholding(yoda_image).show()
   to_gray_double_thresholding(yoda_image).show()
   gray_image = to_greyscale(yoda_image)
   gray_image.show()
   histogramEqualization(gray_image).show()

with Image.open(images[1]) as road_image:
    # gray_image = to_greyscale(road_image)
    gray_image = road_image.convert('L')
    gray_image.show()
    meanFilter(gray_image).show()
