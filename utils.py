import os
import numpy as np
import functools
import keras
import time
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import base64

def image_transpose_exif(im):
    """
    Apply Image.transpose to ensure 0th row of pixels is at the visual
    top of the image, and 0th column is the visual left-hand side.
    Return the original image if unable to determine the orientation.

    As per CIPA DC-008-2012, the orientation field contains an integer,
    1 through 8. Other values are reserved.

    Parameters
    ----------
    im: PIL.Image
       The image to be rotated.
    """

    exif_orientation_tag = 0x0112
    exif_transpose_sequences = [                   # Val  0th row  0th col
        [],                                        #  0    (reserved)
        [],                                        #  1   top      left
        [Image.FLIP_LEFT_RIGHT],                   #  2   top      right
        [Image.ROTATE_180],                        #  3   bottom   right
        [Image.FLIP_TOP_BOTTOM],                   #  4   bottom   left
        [Image.FLIP_LEFT_RIGHT, Image.ROTATE_90],  #  5   left     top
        [Image.ROTATE_270],                        #  6   right    top
        [Image.FLIP_TOP_BOTTOM, Image.ROTATE_90],  #  7   right    bottom
        [Image.ROTATE_90],                         #  8   left     bottom
    ]

    try:
        seq = exif_transpose_sequences[im._getexif()[exif_orientation_tag]]
    except Exception:
        return im
    else:
        return functools.reduce(type(im).transpose, seq, im)

def resize_image(src_image, size=(128,128), bg_color="white"): 
    from PIL import Image, ImageOps 
    
    # resize the image so the longest dimension matches our target size
    src_image.thumbnail(size, Image.ANTIALIAS)
    
    # Create a new square background image
    new_image = Image.new("RGB", size, bg_color)
    
    # Paste the resized image into the center of the square background
    new_image.paste(src_image, (int((size[0] - src_image.size[0]) / 2), int((size[1] - src_image.size[1]) / 2)))
  
    # return the resized image
    return new_image


def load_and_preprocess_image(path_to_image, size, grey=True):
        # open 
        img = Image.open(path_to_image)
        
        # straighten out :
        img = image_transpose_exif(img)
        
        arr = np.array(img)
        
        # Padding with mean of color calculated on the image
        r = int(np.floor(np.mean(arr[:,:,0])))
        g = int(np.floor(np.mean(arr[:,:,1])))
        b = int(np.floor(np.mean(arr[:,:,2])))
        img = resize_image(img, size, bg_color=(r,g,b))
        
        if grey : 
            img = img.convert('L')
            
            # Convert to numpy arrays
            img = np.array(img, dtype=np.float32)

            # Simulate RGB needed to enter pretrained models
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, -1)
            
        else :
            # Convert to numpy arrays
            img = np.array(img, dtype=np.float32)

        # Normalise the images
        img /= 255
        return img


def compile_images(src_folder, size=(128,128)):
    
    # loop through the images
    # Load .jpg only
    filenames = [jpg for jpg in os.listdir(src_folder) if jpg.endswith(".jpg")]
    
    # Define empty arrays where we will store our images and labels
    images = []
            
    for file in filenames[:100]:
        imgFile = os.path.join(src_folder, file)
        img = load_and_preprocess_image(imgFile, size)
        
        # Now we add it to our array
        images.append(img)

    return np.array(images, dtype=np.float32) 
   

def flatten_output(covnet_model, raw_images):
    # Infer the model on raw data
    pred = covnet_model.predict(raw_images)
    # Flatten the prediction array
    flat = pred.reshape(raw_images.shape[0], -1)
    
    return flat

# Function that creates a PCA instance, fits it to the data and returns the instance
def create_fit_PCA(data, n_components=None):
    p = PCA(n_components=n_components, random_state=728)
    p.fit(data)
    return p

# Function to plot the cumulative explained variance of PCA components
def pca_cumsum_plot(pca):
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

# KMEANS
def create_train_kmeans(data, number_of_clusters=3):
    
    k = KMeans(n_clusters=number_of_clusters, random_state=728)

    # Let's do some timings to see how long it takes to train.
    start = time.time()

    # Train it up
    k.fit(data)

    # Stop the timing 
    end = time.time()

    # And see how long that took
    print("Training took {} seconds".format(end-start))
    
    return k

# GAUSSIAN MIXTURE
def create_train_gmm(data, number_of_clusters=3):
    g = GaussianMixture(n_components=number_of_clusters, covariance_type="full", random_state=728)
    
    start=time.time()
    g.fit(data)
    end=time.time()
    
    print("Training took {} seconds".format(end-start))
    
    return g

def reference_labels (nb_clusters):
    labels = []
    dico = {}
    for i in range(nb_clusters):
        dico[i] = []
    return dico

def group_and_plot(src_folder, clusters, nb_clusters, size=(128,128), plot=True, grey=False):
    
    dico = reference_labels(nb_clusters)
    
    filenames = [jpg for jpg in os.listdir(src_folder) if jpg.endswith(".jpg")]
    
    for i, clu in enumerate(clusters):
        dico[clu].append(filenames[i])
        
    if plot :

        figs = []
        
        for clu in dico.keys():

            n_col = 6
            n_row = (len(dico[clu]) // n_col) + 1
            plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
            fig = plt.figure(figsize=(n_col*3 , n_row*3 ))
            
            for i, file in enumerate(dico[clu]):
                imgFile = os.path.join(src_folder, file)
                
                img = load_and_preprocess_image(imgFile, size, grey)
                                
                plt.subplot(n_row, n_col, i + 1, figure=fig)
                plt.imshow(img, figure=fig)

            # fig = plt.show()
            figs.append(fig)
                
    return dico, figs

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


# # Examples
# df = pd.DataFrame({'x': list(range(10)), 'y': list(range(10))})
# st.write(df)

# if st.button('Download Dataframe as CSV'):
#     tmp_download_link = download_link(df, 'YOUR_DF.csv', 'Click here to download your data!')
#     st.markdown(tmp_download_link, unsafe_allow_html=True)

# s = st.text_input('Enter text here')
