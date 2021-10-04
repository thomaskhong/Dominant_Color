from mss import mss
from PIL import Image
import cv2
from sklearn import cluster
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
from mpl_toolkits.mplot3d import axes3d
import numpy as np

def capture_screenshot():
    # Capture entire screen
    with mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        # Convert to PIL/Pillow Image
        return Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')

def resize_screenshot(scale_factor, image):
    baseheight = int(image.size[0]/scale_factor)
    wpercent = (baseheight / float(image.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    resized_image = image.resize((baseheight, hsize), Image.ANTIALIAS)
    return np.array(resized_image)


class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image

    def dominantColors(self):

        #read image
        # img = cv2.imread(self.IMAGE)

        img = self.IMAGE

        #convert to rgb from bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #reshaping to a list of pixels
        img = self.IMAGE.reshape((img.shape[0] * img.shape[1], 3))

        #save image after operation
        self.IMAGE = img

        #use k-means to cluster pixels
        kmeans = KMeans(n_clusters=self.CLUSTERS)
        kmeans.fit(img)

        #assign colors to be the cluster centres
        self.COLORS = kmeans.cluster_centers_

        #save labels
        self.LABELS = kmeans.labels_

        #convert to int from float and return
        return self.COLORS.astype(int)

    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def plotClusters(self):
        fig = plt.figure()
        ax = axes3d.Axes3D(fig)
        for label, pix in zip(self.LABELS, self.IMAGE):
            ax.scatter(pix[0], pix[1], pix[2], color = self.rgb_to_hex(self.COLORS[label]))
        plt.show()

    def plotHistogram(self):
       
        #labels form 0 to no. of clusters
        numLabels = np.arange(0, self.CLUSTERS+1)
       
        #create frequency count tables    
        (hist, _) = np.histogram(self.LABELS, bins = numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()
        
        #appending frequencies to cluster centers
        colors = self.COLORS
        
        #descending order sorting as per frequency count
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()] 
        
        #creating empty chart
        chart = np.zeros((50, 500, 3), np.uint8)
        start = 0
        
        #creating color rectangles
        for i in range(self.CLUSTERS):
            end = start + hist[i] * 500
            
            #getting rgb values
            r = colors[i][0]
            g = colors[i][1]
            b = colors[i][2]
            
            #using cv2.rectangle to plot colors
            cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r,g,b), -1)
            start = end	
        
        #display chart
        plt.figure()
        plt.axis("off")
        plt.imshow(chart)
        plt.show()

img = capture_screenshot()

resize_factor = 8
img = resize_screenshot(resize_factor, img)

clusters = 5

dc = DominantColors(img,clusters)
colors = dc.dominantColors()
dc.plotHistogram()
print(colors)

hsv_colors = pltcol.rgb_to_hsv(colors/255)*255
print(hsv_colors)

