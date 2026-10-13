import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# --- PARAMETRY WIZUALIZACJI ---
SIZE_FACTOR = 50  
MIN_SIZE = 5      
# ------------------------------

def load_data(input_path, output_path):
    masses = []
    initial_pos = []
    
    # 1. Wczytanie danych z input.txt (Masy i T=0)
    with open(input_path, 'r') as f:
        lines = f.readlines()
        nbodies = int(lines[0].strip())
        for i in range(1, nbodies + 1):
            parts = list(map(float, lines[i].split()))
            initial_pos.append(parts[:3])
            masses.append(parts[6])
            
    masses = np.array(masses)
    sizes = (masses**(1/3)) * SIZE_FACTOR + MIN_SIZE
    sun_idx = np.argmax(masses) # Indeks Słońca

    # 2. Wczytanie kroków z output.txt
    steps = [np.array(initial_pos)]
    try:
        with open(output_path, 'r') as f:
            current_step_data = []
            for line in f:
                line = line.strip()
                if not line or "PHYS" in line: continue
                if line.startswith("Step"):
                    if current_step_data:
                        steps.append(np.array(current_step_data))
                        current_step_data = []
                else:
                    parts = list(map(float, line.split()))
                    if len(parts) >= 3:
                        current_step_data.append(parts[:3])
            if current_step_data:
                steps.append(np.array(current_step_data))
    except FileNotFoundError:
        print("Nie znaleziono pliku output.txt - tylko klatka 0")

    return steps, sizes, sun_idx

def animate_simulation(input_file, output_file):
    steps, sizes, sun_idx = load_data(input_file, output_file)
    total_frames = len(steps)
    
    # Przygotowanie danych relatywnych względem Słońca
    relative_steps = [s - s[sun_idx] for s in steps]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['gold' if i == sun_idx else 'deepskyblue' for i in range(len(sizes))]

    # Ustawienie granic na podstawie najdalszej planety
    all_rel_data = np.concatenate(relative_steps)
    max_val = np.max(np.abs(all_rel_data)) * 1.1
    
    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])

    # Inicjalizacja klatek
    scat = ax.scatter(relative_steps[0][:, 0], relative_steps[0][:, 1], relative_steps[0][:, 2], 
                      s=sizes, c=colors, edgecolors='white', linewidth=0.5, alpha=0.8)

    # Widok z góry
    ax.view_init(elev=90, azim=-90)

    def update(frame):
        # Aktualizacja pozycji punktów
        rel_pos = relative_steps[frame]
        scat._offsets3d = (rel_pos[:, 0], rel_pos[:, 1], rel_pos[:, 2])
        
        # AKTUALIZACJA LICZNIKA W TYTULE
        ax.set_title(f'Symulacja N-Body | Klatka: {frame + 1} / {total_frames}\n(Centrum: Najcięższy obiekt)')
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=total_frames, 
                                  interval=10, blit=False)
    plt.show()

if __name__ == "__main__":
    animate_simulation('input2.txt', 'output.txt')
    #animate_simulation('tests/in_10.txt', 'tests/out_10.txt')