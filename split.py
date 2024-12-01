import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import faiss
import numpy as np
import h5py
import pickle
import os
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from netvlad import NetVLAD

class ImageSearcher:
    def __init__(self, chunks_dirs=['database_chunks_part1', 'database_chunks_part2']):
        """
        Khởi tạo Image Searcher
        Args:
            chunks_dirs (list): Danh sách các đường dẫn đến thư mục chứa database chunks
        """
        print("Initializing model...")
        # Khởi tạo VGG16 model
        self.encoder = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.encoder = torch.nn.Sequential(*list(self.encoder.features.children()))
        
        # Khởi tạo NetVLAD layer
        self.net_vlad = NetVLAD(num_clusters=64, dim=512)
        self.model = torch.nn.Sequential(self.encoder, self.net_vlad)
        
        # Chuyển model sang GPU nếu có
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transform cho ảnh đầu vào
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load database từ nhiều thư mục
        print("Loading database chunks...")
        self.load_database_chunks(chunks_dirs)
        print(f"Using device: {self.device}")
    
    def load_database_chunks(self, chunks_dirs):
        """
        Load và ghép các chunks của database
        Args:
            chunks_dirs (list): Danh sách các đường dẫn đến thư mục chứa chunks
        """
        self.features = []
        self.image_paths = []
        
        # Duyệt qua từng thư mục
        for chunks_dir in chunks_dirs:
            # Lấy danh sách các chunk files trong thư mục
            chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith('.h5')]
            chunk_files.sort()
            
            print(f"\nLoading chunks from {chunks_dir}")
            print(f"Found {len(chunk_files)} chunks")
            
            for chunk_file in chunk_files:
                chunk_num = chunk_file.split('.')[0].split('_')[1]
                print(f"Loading chunk {chunk_num}...")
                
                # Load features
                with h5py.File(os.path.join(chunks_dir, f'chunk_{chunk_num}.h5'), 'r') as f:
                    chunk_features = f['features'][:]
                    self.features.append(chunk_features)
                
                # Load paths
                with open(os.path.join(chunks_dir, f'chunk_{chunk_num}_paths.pkl'), 'rb') as f:
                    chunk_paths = pickle.load(f)
                    self.image_paths.extend(chunk_paths)
        
        # Ghép các features lại
        self.features = np.concatenate(self.features, axis=0)
        
        # Chuẩn hóa features
        print("Normalizing features...")
        self.normalized_features = self.features / (np.linalg.norm(self.features, axis=1, keepdims=True) + 1e-8)
        
        print(f"\nLoaded total {len(self.image_paths)} images")
        print(f"Feature shape: {self.features.shape}")
    
    def search(self, query_path, top_k=5):
        """
        Tìm kiếm ảnh tương tự
        Args:
            query_path (str): Đường dẫn đến ảnh query
            top_k (int): Số lượng kết quả trả về
        Returns:
            list: Danh sách các kết quả tìm kiếm
        """
        # Kiểm tra file tồn tại
        if not os.path.exists(query_path):
            raise FileNotFoundError(f"Query image not found: {query_path}")
        
        # Load và xử lý ảnh query
        query_img = Image.open(query_path).convert('RGB')
        query_tensor = self.transform(query_img)
        query_tensor = query_tensor.unsqueeze(0).to(self.device)
        
        # Trích xuất features
        with torch.no_grad():
            query_features = self.model(query_tensor).cpu().numpy()
        
        # Chuẩn hóa query features
        query_features = query_features.squeeze()
        query_features = query_features / (np.linalg.norm(query_features) + 1e-8)
        
        # Tính cosine similarity
        similarities = np.dot(self.normalized_features, query_features)
        
        # Chuyển sang khoảng cách
        distances = 1 - similarities
        
        # Thêm nhiễu nhỏ để tránh trường hợp bằng nhau
        distances += np.random.normal(0, 1e-6, distances.shape)
        
        # Lấy top-k kết quả
        indices = np.argsort(distances)[:top_k]
        results = []
        for idx in indices:
            results.append({
                'path': self.image_paths[idx],
                'distance': float(distances[idx])
            })
        
        return results

    def visualize_results(self, query_path, results):
        """
        Hiển thị ảnh query và các kết quả
        Args:
            query_path (str): Đường dẫn đến ảnh query
            results (list): Danh sách kết quả từ hàm search
        """
        # Đọc ảnh query
        query_img = cv2.imread(query_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # Tạo subplot
        plt.figure(figsize=(20, 4))
        
        # Hiển thị ảnh query
        plt.subplot(1, len(results) + 1, 1)
        plt.imshow(query_img)
        plt.title('Query Image')
        plt.axis('off')
        
        # Hiển thị kết quả
        for i, result in enumerate(results):
            try:
                img = cv2.imread(result['path'])
                if img is None:
                    raise Exception(f"Could not load image: {result['path']}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                plt.subplot(1, len(results) + 1, i + 2)
                plt.imshow(img)
                plt.title(f'Distance: {result["distance"]:.2f}')
                plt.axis('off')
            except Exception as e:
                print(f"Error displaying result {i+1}: {str(e)}")
        
        plt.tight_layout()
        plt.show()

def main():
    """Hàm chính để chạy chương trình"""
    try:
        # Khởi tạo searcher với thư mục chứa chunks
        searcher = ImageSearcher(['database_chunks_part1', 'database_chunks_part2'])
        
        # Tìm kiếm với một ảnh query
        query_path = "R.jpg"  # Thay đổi đường dẫn này theo ảnh query của bạn
        results = searcher.search(query_path, top_k=5)
        
        # Hiển thị kết quả
        searcher.visualize_results(query_path, results)
        
        # In thông tin kết quả
        print("\nSearch Results:")
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Path: {result['path']}")
            print(f"Distance: {result['distance']:.4f}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()