from base64 import b64encode
from flask import Flask, render_template, jsonify, request
from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np
import queue
import threading

# ------------ CONSTANTS ------------
UPLOAD_FOLDER = 'static/media'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000

q = queue.Queue()

# ------------ BACK END ------------
@dataclass
class ImageWorker():
    img: cv2.Mat = None
    initial_height: int = 0
    initial_width: int = 0
    
    current_height: int = 0
    current_width: int = 0
    
    curr_scaling_factor: int = 0
    prev_scaling_factor: int = 0
    
    seam_cache = {}
    
    def set_scaling_factor(self, prev, curr):
        self.prev_scaling_factor = prev
        self.curr_scaling_factor = curr
    
    def get_user_facing_image(self) -> cv2.Mat:
        return self.img
    
    def resize(self, img: cv2.Mat):
        h, w, _ = img.shape
        
        limit = 666
        
        if w <= limit and h <= limit:
            return img

        max_dim = max(w, h)        
        ratio = max_dim / limit
        return cv2.resize(img, (int(w / ratio), int(h / ratio)))

    def instantiate(self, file) -> None:
        
        image_data = np.frombuffer(file.read(), dtype=np.uint8)
        img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        resized = self.resize(img)
        self.img = resized
        
        # set dimensions
        self.current_height, self.current_width, _ = self.img.shape
        self.initial_height, self.initial_width, _ = self.img.shape
        
        self.set_scaling_factor(self.current_width, self.current_width)
    
    def process(self, scaling_factor: int) -> None:
        
        new_dim = (self.current_height, self.curr_scaling_factor)
                
        if self.curr_scaling_factor > self.prev_scaling_factor:
            # Stretching the image - seam will be cached, no need to re-process
            self.img = self.add_seam(scaling_factor)
            return
        elif new_dim in self.seam_cache:
            # Seam is already cached, no need to re-process
            derived_seam = list(map(lambda x: x[0][1] if x else 0, self.seam_cache[new_dim]))
            self.img = self.remove_seam(derived_seam)
            return
                
        # grayscale
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        
        # calculate sobel convolution
        sobel_map = self.sobel(gray)
        
        # find min-energy path in image
        energy_data = self.calculate_energies(sobel_map)
        
        # define the points along the seam
        vertical_seam = self.define_path(energy_data)
        
        #self.prev_scaling_factor = scaling_factor
        self.img = self.remove_seam(vertical_seam)
        
    def add_seam(self, scaling_factor):
        height, width, channels = self.img.shape
        path_data = self.seam_cache[(height, scaling_factor)]
        new_image = np.zeros((height, width + 1, channels), dtype=self.img.dtype)
        
        for row in range(self.current_height - 1):
            loc, pixel = path_data[row][0], path_data[row][1]
            r, c = loc
            new_image[row, :c] = self.img[row, :c]
            new_image[row, c] = pixel
            new_image[row, c + 1:] = self.img[row, c:]

        
        self.current_height, self.current_width, _ = new_image.shape
        return new_image
    
    def sobel(self, gray_img):
        sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.magnitude(sobel_x, sobel_y)
    
    def calculate_energies(self, grid: np.ndarray) -> np.ndarray:
        for row in reversed(range(self.current_height - 1)):
            min_energy_row = np.minimum(grid[row + 1, :-2], grid[row + 1, 1:-1])
            min_energy_row = np.minimum(min_energy_row, grid[row + 1, 2:])
            min_energy_row_padded = np.pad(min_energy_row, (1, 1), mode='edge')
            grid[row] += min_energy_row_padded

        return grid

    def define_path(self, numbers):
        """
        Ensure that this function prefers to take a seam from seam_cache rather than calculate a new one.
        If the user shrinks then stretches an image, then shrinks it again, use the cached data if it exists.
        """
        min_point_layer = np.zeros(self.current_height, np.int32)
        pixel_layer = np.empty(self.current_height, dtype=object)
        
        min_point_layer[0] = np.argmin(numbers[0])
        
        col = min_point_layer[0]

        pixel_layer[0] = [(0, col), self.img[0][col]]
        
        for row in range(1, self.current_height - 1):
            next_x = col
            for k in range(max(0, col - 1), min(self.current_width, col + 2)):
                if numbers[row][k] < numbers[row][next_x]:
                    next_x = k
            col = next_x
            min_point_layer[row] = col
            pixel_layer[row] = [(row, col), self.img[row][col]] 

        # width might need to be changed!
        self.seam_cache[(self.current_height, self.current_width)] = pixel_layer
        return min_point_layer

    def remove_seam(self, seam):
        height, width, channels = self.img.shape
        path_points = np.array(seam)
        new_image = np.zeros((height, width - 1, channels), dtype=self.img.dtype)
        
        for row in range(self.current_height):
            point_to_remove = path_points[row]
            new_image[row, :point_to_remove] = self.img[row, :point_to_remove]
            new_image[row, point_to_remove:] = self.img[row, point_to_remove + 1:]
        
        self.current_height, self.current_width, _ = new_image.shape
        return new_image
    
iw = ImageWorker()

def send_image_to_front_end(img: cv2.Mat):
    
    success, encoded_image = cv2.imencode('.png', img)
    
    if not success:
        return jsonify({'message': 'Error encoding the image'}), 500
    
    base64_image: bytes = b64encode(encoded_image).decode('utf-8')
    return jsonify({'message': 'Image processed successfully', 'image': base64_image}), 200    

def add_to_queue():
    while True:
        scaling_factor = q.get()
        if scaling_factor is None:
            break
        iw.set_scaling_factor(iw.curr_scaling_factor, scaling_factor)
        iw.process(scaling_factor)
        q.task_done() 
# ------------ FRONT END ------------
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/receive_image', methods=['POST'])
def receive_image():
    if 'picture' not in request.files:
        return 'No image file found in the request', 400
    
    iw.instantiate(request.files['picture'])
    threading.Thread(target=add_to_queue, daemon=True).start()
    return send_image_to_front_end(iw.get_user_facing_image())
    
@app.route('/process_image', methods=['POST'])
def process_image():
    st = perf_counter()    
    
    scaling_factor: int = int(request.form['scaling_factor'])
    
    q.put(scaling_factor)
    
    end = perf_counter()
    # print(f'COMPLETED IN {end - st} SECONDS.')
    return send_image_to_front_end(iw.get_user_facing_image())
    

if __name__ == '__main__':
    """
    TODO:
    - ✅ Implement a blocking cache that waits until all requests are complete to begin a new request
    - ✅ Save deleted seams to enable increasing width
    - ✅ Allow the user to either shrink or grow the image
    
    - OPTIMIZE...
    
    """
    app.run(debug=True, port=8000)