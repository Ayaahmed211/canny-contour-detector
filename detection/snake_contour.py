import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend FIRST
import numpy as np
import cv2
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from skimage.filters import sobel, gaussian
from skimage.util import img_as_float

class GreedySnake:
    def __init__(self, image, initial_contour, alpha=0.3, beta=0.5, gamma=1.5,
                 max_iterations=200, convergence_threshold=0.5):
        
        self.image = image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
            
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        self.edge_map = self._compute_edge_map_enhanced()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        self.contour = np.array(initial_contour, dtype=np.float32)
        self.num_points = len(self.contour)
        
        self.contour_energy = []
        self.convergence_history = []
        
        # NEW: Store history to visualize the evolution
        self.contour_history = [np.copy(self.contour)]


    def _resample_contour(self, min_dist=2.0, max_dist=15.0):
        """
        Dynamically inserts or removes points to maintain even spacing.
        Prevents contour bunching and edge leakage.
        """
        new_contour = [self.contour[0]]
        
        for i in range(1, len(self.contour)):
            prev_pt = new_contour[-1]
            curr_pt = self.contour[i]
            
            dist = np.linalg.norm(curr_pt - prev_pt)
            
            if dist > max_dist:
                # Contour is stretching too thin: Insert a midpoint
                midpoint = (prev_pt + curr_pt) / 2.0
                new_contour.append(midpoint)
                new_contour.append(curr_pt)
            elif dist < min_dist:
                # Points are bunching up: Drop the current point
                continue
            else:
                # Distance is fine: Keep the point
                new_contour.append(curr_pt)
                
        # Close the loop check for the last and first point
        dist_end = np.linalg.norm(new_contour[-1] - new_contour[0])
        if dist_end < min_dist and len(new_contour) > 3:
            new_contour.pop()
            
        self.contour = np.array(new_contour, dtype=np.float32)
        self.num_points = len(self.contour)

    def _compute_edge_map_enhanced(self):
        """
        Computes an enhanced edge map using CLAHE and Distance Transform.
        This massively increases the 'capture range' of the snake.
        """
        # 1. Local Contrast Enhancement
        # Helps find weak edges in shadows or bright spots
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(self.gray)
        
        # 2. Denoising
        # Bilateral filter is better than Gaussian here because it preserves edges 
        # while smoothing out flat regions/noise.
        blurred = cv2.bilateralFilter(enhanced_gray, d=9, sigmaColor=75, sigmaSpace=75)
        
        # 3. Hard Edge Detection (Canny)
        # Using Otsu's thresholding to dynamically find the best Canny thresholds
        high_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_thresh = 0.5 * high_thresh
        edges = cv2.Canny(blurred, low_thresh, high_thresh)
        
        # 4. Distance Transform (The Secret Weapon)
        # We want the energy to be 0 at the edge, and increase the further away we get.
        # cv2.distanceTransform measures distance to the nearest ZERO pixel.
        # So, we invert the Canny edges (edges become 0, background becomes 255).
        inverted_edges = cv2.bitwise_not(edges)
        dist_transform = cv2.distanceTransform(inverted_edges, cv2.DIST_L2, 5)
        
        # 5. Normalize to [0.0, 1.0]
        # This acts as our external energy landscape. The snake will slide down 
        # this distance gradient until it hits the 0-energy edge.
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        
        return dist_transform
    
    def _find_neighborhood_minimum(self, point_idx, avg_dist, window_size=5):
            """Vectorized Greedy Search for O(1) local minimization"""
            prev_pt = self.contour[(point_idx - 1) % self.num_points]
            curr_pt = self.contour[point_idx]
            next_pt = self.contour[(point_idx + 1) % self.num_points]
            
            half_window = window_size // 2
            
            # Generate a grid of all 25 candidate points
            dx, dy = np.meshgrid(np.arange(-half_window, half_window + 1), 
                                np.arange(-half_window, half_window + 1))
            candidates = curr_pt + np.column_stack((dx.ravel(), dy.ravel()))
            
            # Filter candidates that fall outside image boundaries
            valid_mask = (candidates[:, 0] >= 0) & (candidates[:, 0] < self.edge_map.shape[1]) & \
                        (candidates[:, 1] >= 0) & (candidates[:, 1] < self.edge_map.shape[0])
            valid_candidates = candidates[valid_mask]
            
            if len(valid_candidates) == 0:
                return curr_pt, 0.0 # Failsafe
                
            # 1. Continuity Energy (Maintain average spacing)
            # Using |avg_dist - dist_to_prev| prevents the rubber-band collapse
            dists = np.linalg.norm(valid_candidates - prev_pt, axis=1)
            e_cont = np.abs(avg_dist - dists)
            if e_cont.max() > 0: e_cont /= e_cont.max()
            
            # 2. Curvature Energy (Minimize 2nd derivative)
            # |prev - 2*candidate + next|^2
            e_curv = np.linalg.norm(prev_pt - 2 * valid_candidates + next_pt, axis=1)**2
            if e_curv.max() > 0: e_curv /= e_curv.max()
            
            # 3. External Energy (Image Edges)
            coords = np.round(valid_candidates).astype(int)
            
            # ---> THIS IS THE FIX <---
            # Mathematically clip coordinates so they never exceed the image bounds
            coords[:, 0] = np.clip(coords[:, 0], 0, self.edge_map.shape[1] - 1)
            coords[:, 1] = np.clip(coords[:, 1], 0, self.edge_map.shape[0] - 1)
            # -------------------------
            
            e_ext = self.edge_map[coords[:, 1], coords[:, 0]]
            if e_ext.max() > e_ext.min(): 
                e_ext = (e_ext - e_ext.min()) / (e_ext.max() - e_ext.min() + 1e-6)
                
            # Total Energy
            total_energy = self.alpha * e_cont + self.beta * e_curv + self.gamma * e_ext
            
            # Find best candidate
            best_idx = np.argmin(total_energy)
            return valid_candidates[best_idx], total_energy[best_idx]
    def evolve(self):
        """Evolve snake using greedy algorithm"""
        print("Starting vectorized snake evolution...")
        
        for iteration in range(self.max_iterations):
            total_movement = 0
            iteration_energy = []
            
            diffs = self.contour - np.roll(self.contour, 1, axis=0)
            avg_dist = np.mean(np.linalg.norm(diffs, axis=1))
            
            for i in range(self.num_points):
                best_point, min_energy = self._find_neighborhood_minimum(i, avg_dist, window_size=5)
                
                movement = np.linalg.norm(best_point - self.contour[i])
                total_movement += movement
                
                self.contour[i] = best_point
                iteration_energy.append(min_energy)
            
            avg_energy = np.mean(iteration_energy)
            self.contour_energy.append(avg_energy)
            
            avg_movement = total_movement / self.num_points
            self.convergence_history.append(avg_movement)
            
            # Snapshots for visualization
            if iteration % 5 == 0:  
                self._resample_contour()
                # Save a snapshot of the contour state after resampling
                self.contour_history.append(np.copy(self.contour))
            
            if avg_movement < self.convergence_threshold:
                print(f"Converged after {iteration} iterations")
                break
                
        # Ensure the absolute final state is saved
        self.contour_history.append(np.copy(self.contour))
        return self.contour

    
    def get_chain_code(self):
        """Generate Freeman chain code"""
        chain_code = []
        n = len(self.contour)
        
        # Direction mapping (Freeman chain code: 0-7)
        directions = {
            (0, 1): 0,   # Right
            (1, 1): 1,   # Down-Right
            (1, 0): 2,   # Down
            (1, -1): 3,  # Down-Left
            (0, -1): 4,  # Left
            (-1, -1): 5, # Up-Left
            (-1, 0): 6,  # Up
            (-1, 1): 7   # Up-Right
        }
        
        for i in range(n):
            p1 = self.contour[i]
            p2 = self.contour[(i + 1) % n]
            
            # Calculate direction
            dx = int(round(p2[0] - p1[0]))
            dy = int(round(p2[1] - p1[1]))
            
            # Find closest direction
            if dx != 0 or dy != 0:
                best_dir = None
                min_dist = float('inf')
                
                for (ddx, ddy), code in directions.items():
                    dist = np.sqrt((dx - ddx)**2 + (dy - ddy)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_dir = code
                
                if best_dir is not None:
                    chain_code.append(best_dir)
        
        return chain_code
    
    def compute_perimeter(self):
        """Compute perimeter of contour"""
        perimeter = 0
        n = len(self.contour)
        
        for i in range(n):
            p1 = self.contour[i]
            p2 = self.contour[(i + 1) % n]
            perimeter += np.linalg.norm(p2 - p1)
        
        return perimeter
    
    def compute_area(self):
        """Compute area inside contour using shoelace formula"""
        x = self.contour[:, 0]
        y = self.contour[:, 1]
        
        # Shoelace formula
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area
    
    def get_visualization(self):
        """Create a time-lapse visualization showing how the contour evolved."""
        # Exact colors from your React UI Theme
        bg_color = '#0d1120'
        text_color = '#d4dff2'
        grid_color = '#2a3550'
        orange = '#ff6b35'
        cyan = '#38bdf8'
        green = "#e00d0d"

        # 1x3 Layout: Time-lapse, Edge Map, Energy Convergence
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor=bg_color)
        
        # Helper to style the axes consistently
        for ax in axes:
            ax.set_facecolor(bg_color)
            ax.tick_params(colors=text_color)
            for spine in ax.spines.values():
                spine.set_color(grid_color)
                spine.set_linewidth(1.5)

        # --- Subplot 1: Time-lapse Evolution ---
        axes[0].axis('off')
        if len(self.image.shape) == 3:
            axes[0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(self.image, cmap='gray')
            
        axes[0].set_title('Contour Evolution (Time-lapse)', color=text_color, pad=15, fontsize=15, fontweight='bold')

        # Draw the History (The Time-lapse Effect)
        num_snapshots = len(self.contour_history)
        for i, hist_contour in enumerate(self.contour_history):
            contour_closed = np.vstack([hist_contour, hist_contour[0]])
            
            if i == 0:
                axes[0].plot(contour_closed[:, 0], contour_closed[:, 1], color=orange, linewidth=2.5, linestyle='--', label='Initial')
            elif i == num_snapshots - 1:
                axes[0].plot(contour_closed[:, 0], contour_closed[:, 1], color=green, linewidth=3, label='Final')
            else:
                progress = i / num_snapshots
                axes[0].plot(contour_closed[:, 0], contour_closed[:, 1], color=cyan, linewidth=1.5, alpha=0.1 + (0.4 * progress))
                
        axes[0].legend(loc='upper right', facecolor=bg_color, edgecolor=grid_color, labelcolor=text_color)


        # --- Subplot 2: Edge Map & Target ---
        axes[1].axis('off')
        axes[1].set_title('Energy Field (Dark = Target Edge)', color=text_color, pad=15, fontsize=15, fontweight='bold')
        
        # Using viridis makes the "valleys" (edges) dark purple and the "hills" bright yellow
        im = axes[1].imshow(self.edge_map, cmap='viridis')
        
        # Plot the final contour on the edge map to see it resting in the 'valley'
        final_contour = self.contour_history[-1]
        final_closed = np.vstack([final_contour, final_contour[0]])
        axes[1].plot(final_closed[:, 0], final_closed[:, 1], color=green, linewidth=2.5)
        
        # Add a colorbar to explain what the colors mean
        cbar = fig.colorbar(im, ax=axes[1], shrink=0.7, pad=0.05)
        cbar.ax.tick_params(colors=text_color)
        cbar.set_label('Distance to Edge (Energy)', color=text_color, size=12)


        # --- Subplot 3: Energy Convergence ---
        if self.contour_energy:
            iterations = range(len(self.contour_energy))
            
            axes[2].fill_between(iterations, self.contour_energy, color=orange, alpha=0.15)
            axes[2].plot(iterations, self.contour_energy, color=orange, linewidth=2.5)
            axes[2].plot(iterations[-1], self.contour_energy[-1], 'o', color=cyan, markersize=6)
            
            axes[2].set_xlabel('Iteration', color=text_color, fontsize=12, fontweight='500')
            axes[2].set_ylabel('Average Energy', color=text_color, fontsize=12, fontweight='500')
            axes[2].set_title('Energy Convergence', color=text_color, pad=15, fontsize=15, fontweight='bold')
            axes[2].grid(True, color=grid_color, linestyle='--', alpha=0.7)
            
            axes[2].spines['top'].set_visible(False)
            axes[2].spines['right'].set_visible(False)

        plt.tight_layout(pad=2.0)

        # Convert plot to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor=bg_color, edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return image_base64