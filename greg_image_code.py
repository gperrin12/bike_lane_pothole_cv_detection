import scipy.ndimage as nd

img = nd.imread("bike_lane_example.jpg")
imshow(img)
img = img[1000:]
clf()
imshow(img)
gf = nd.filters.gaussian_filter
img.shape
red, grn, blu = img.transpose(2, 0, 1)
clf()
imhow(red)
imshow(red)
imshow(grn)
imshow(blu)
wind = (red > 200) & (grn > 200) & (blu > 200)
imshow(wind)
imshow(gf(1.0 * wind))
imshow(gf(1.0 * wind), 5)
imshow(gf(1.0 * wind, 5))
wind = (red > 150) & (grn > 150) & (blu > 150)
imshow(wind)
imshow(gf(1.0 * wind, 5))
imshow(gf(1.0 * wind, 20))
imshow(gf(1.0 * wind, 40))
imshow(gf(1.0 * wind, 40) > 10)
imshow(gf(1.0 * wind, 40) > 30)
imshow(gf(1.0 * wind, 40) > 100)
imshow(gf(1.0 * wind, 40) > 5)
imshow(gf(1.0 * wind, 40) > 1)
imshow(gf(1.0 * wind, 40))
gf(1.0 * wind, 40)[690, 680]
clf()
imshow(gf(1.0 * wind, 40) > 0.08)
img_L = img.mean(-1)
imshow(img_L)
img_L = gf(img_L, 2)
img_L = img.mean(-1)
img_Lsm = gf(img_L, 2)
gd = np.abs((img_Lsm[2:, 2:] - img_Lsm[:-2, 2:]) + (img_Lsm[2:, 2:] - img_Lsm[2:, :-2]))
imshow(gd)
imshow(gd > 1)
imshow(gd > 10)
imshow(gd > 20)
imshow(gd > 30)
edges = gd > 30
imshow(gf(1.0 * wind, 40) > 0.08)
wreg = gf(1.0 * wind, 40) > 0.08
edges.shape
wreg.shape
imshow(edges)
imshow(edges*wreg[2:, 2:])
imshow(edges)
imshow(edges*wreg[2:, 2:])
