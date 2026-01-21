import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.constants import SKELETON_CONNECTIONS, TORSO_CONNECTIONS, UPPER_BODY_CONNECTIONS
import json


def _detect_skeleton_type(keypoints):
    """
    Rileva automaticamente il tipo di scheletro in base al numero di keypoints.
    
    Args:
        keypoints: array (N, 2) con coordinate
    
    Returns:
        tuple: (skeleton_type, connections)
               skeleton_type: 'full', 'torso', 'upper_body'
    """
    n_points = len(keypoints)
    
    if n_points == 17:
        return 'full', SKELETON_CONNECTIONS
    elif n_points == 13:
        return 'torso', TORSO_CONNECTIONS
    elif n_points == 8:
        return 'upper_body', UPPER_BODY_CONNECTIONS
    else:
        # Fallback: nessuna connessione, solo punti
        return 'unknown', []


def _prepare_keypoints(pose):
    """
    Converte pose in formato keypoints (N, 2).
    
    Args:
        pose: array flat (2N,) oppure array (N, 2)
    
    Returns:
        array (N, 2) con coordinate
    """
    if len(pose.shape) == 1:
        # Flat array: reshape
        n_points = len(pose) // 2
        return pose.reshape(n_points, 2)
    else:
        return pose


def _draw_skeleton(ax, keypoints, connections, point_color='red', line_color='blue',
                   point_size=50, line_width=2.0, point_alpha=0.8, line_alpha=0.7):
    """
    Disegna uno scheletro su un asse matplotlib.
    
    Args:
        ax: matplotlib axis
        keypoints: array (N, 2) con coordinate
        connections: lista di tuple (start_idx, end_idx)
        point_color: colore dei keypoints
        line_color: colore delle connessioni
        point_size: dimensione dei punti
        line_width: spessore delle linee
        point_alpha: trasparenza punti
        line_alpha: trasparenza linee
    """
    # Disegna connessioni
    for start_idx, end_idx in connections:
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start = keypoints[start_idx]
            end = keypoints[end_idx]
            
            # Skip se punto non valido (troppo vicino a zero)
            if not (abs(start[0]) < 1e-5 and abs(start[1]) < 1e-5) and \
               not (abs(end[0]) < 1e-5 and abs(end[1]) < 1e-5):
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       color=line_color, linewidth=line_width, alpha=line_alpha)
    
    # Disegna keypoints
    valid_points = keypoints[~np.all(np.abs(keypoints) < 1e-5, axis=1)]
    
    if len(valid_points) > 0:
        ax.scatter(valid_points[:, 0], valid_points[:, 1], 
                  c=point_color, s=point_size, zorder=5, alpha=point_alpha)


def plot_skeleton(pose, title="Skeleton", show=True, skeleton_type=None):
    """
    Plotta uno scheletro, rilevando automaticamente il tipo.
    
    Args:
        pose: array flat (2N,) oppure array (N, 2)
        title: titolo del plot
        show: se True mostra il plot, altrimenti ritorna fig, ax
        skeleton_type: se specificato ('full', 'torso', 'upper_body'), forza il tipo
    
    Returns:
        Se show=False, ritorna (fig, ax)
    """
    keypoints = _prepare_keypoints(pose)
    
    # Rileva tipo di scheletro
    if skeleton_type is None:
        detected_type, connections = _detect_skeleton_type(keypoints)
    else:
        if skeleton_type == 'full':
            connections = SKELETON_CONNECTIONS
        elif skeleton_type == 'torso':
            connections = TORSO_CONNECTIONS
        elif skeleton_type == 'upper_body':
            connections = UPPER_BODY_CONNECTIONS
        else:
            connections = []
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    _draw_skeleton(ax, keypoints, connections)
    
    # Imposta limiti e aspetto
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Inverte Y per avere la testa in alto
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
    if show:
        plt.tight_layout()
        plt.show()
    else:
        return fig, ax

def plot_codebook(codebook, title="Pose Codebook", show=True, skeleton_type=None, polar=False):
    """
    Plotta tutti i centroidi del codebook in una griglia.
    
    Args:
        codebook: PoseCodebook con centroids
        title: titolo del plot
        show: se True mostra il plot, altrimenti ritorna fig, axes
        skeleton_type: se specificato ('full', 'torso', 'upper_body'), forza il tipo
    
    Returns:
        Se show=False, ritorna (fig, axes)
    """
    n_clusters = codebook.n_clusters
    
    # Calcola dimensioni griglia
    cols = min(8, n_clusters)
    rows = (n_clusters + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    
    # Gestisci caso con singola riga/colonna
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx in range(n_clusters):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        centroid = codebook.poses[idx]
        token = codebook.tokens[idx]
        
        # Prepara keypoints
        keypoints = _prepare_keypoints(centroid)
        
        # Rileva tipo di scheletro
        if skeleton_type is None:
            detected_type, connections = _detect_skeleton_type(keypoints)
        else:
            if skeleton_type == 'full':
                connections = SKELETON_CONNECTIONS
            elif skeleton_type == 'torso':
                connections = TORSO_CONNECTIONS
            elif skeleton_type == 'upper_body':
                connections = UPPER_BODY_CONNECTIONS
            else:
                connections = []
        
        # Disegna scheletro usando la funzione comune
        _draw_skeleton(ax, keypoints, connections, 
                      point_size=20, line_width=1.5)
        
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f"Token {token}", fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)
    
    # Nascondi assi vuoti
    for idx in range(n_clusters, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    fig.suptitle(title, fontsize=14, y=0.995)
    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        return fig, axes


def animate_skeleton_from_dataset(sample, codebook, skeleton_type=None, 
                                   fps=30, save_path=None, show=True):
    """
    Anima lo scheletro dato un indice del dataset e il codebook.
    
    Args:
        sample: dizionario con i dati del sample
        codebook: PoseCodebook per decodificare i token
        skeleton_type: tipo di scheletro ('full', 'torso', 'upper_body')
        fps: frame per secondo dell'animazione (default 30)
        save_path: se specificato, salva l'animazione in questo path (es. 'animation.gif')
        show: se True mostra l'animazione, altrimenti ritorna solo l'oggetto
    
    Returns:
        animation: oggetto FuncAnimation di matplotlib
    """
    
    
    # Estrai i token
    tokens = np.array(sample['tokens'])
    text = sample.get('text', 'No text')
    n_frames = len(tokens)
    
    # Decodifica i token usando il codebook per ottenere le pose
    poses = []
    for token in tokens:
        # Trova il centroide corrispondente al token
        token_idx = np.where(codebook.tokens == token)[0]
        if len(token_idx) > 0:
            pose = codebook.poses[token_idx[0]]
            poses.append(pose)
        else:
            # Token non trovato, usa posa vuota
            poses.append(np.zeros_like(codebook.poses[0]))
    
    poses = np.array(poses)
    
    # Prepara il primo frame
    first_pose = _prepare_keypoints(poses[0])
    
    # Rileva tipo di scheletro
    if skeleton_type is None:
        detected_type, connections = _detect_skeleton_type(first_pose)
    else:
        if skeleton_type == 'full':
            connections = SKELETON_CONNECTIONS
        elif skeleton_type == 'torso':
            connections = TORSO_CONNECTIONS
        elif skeleton_type == 'upper_body':
            connections = UPPER_BODY_CONNECTIONS
        else:
            connections = []
    
    # Calcola i limiti per tenere tutto in vista
    all_keypoints = np.vstack([_prepare_keypoints(p) for p in poses])
    valid_points = all_keypoints[~np.all(np.abs(all_keypoints) < 1e-5, axis=1)]
    
    if len(valid_points) > 0:
        x_min, y_min = valid_points.min(axis=0)
        x_max, y_max = valid_points.max(axis=0)
        margin = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= margin * x_range
        x_max += margin * x_range
        y_min -= margin * y_range
        y_max += margin * y_range
    else:
        x_min, x_max = -1, 1
        y_min, y_max = -1, 1
    
    # Crea la figura
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Invertito per avere la testa in alto
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Elementi che verranno aggiornati
    lines = []
    points = None
    
    # Testo per mostrare frame e testo
    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                         verticalalignment='top', fontsize=12,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    caption_text = ax.text(0.5, 0.02, text, transform=ax.transAxes,
                          horizontalalignment='center', verticalalignment='bottom',
                          fontsize=10, wrap=True,
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    def init():
        """Inizializza l'animazione."""
        return []
    
    def update(frame):
        """Aggiorna l'animazione per ogni frame."""
        nonlocal lines, points
        
        # Rimuovi elementi precedenti
        for line in lines:
            line.remove()
        lines = []
        
        if points is not None:
            points.remove()
        
        # Ottieni keypoints del frame corrente
        keypoints = _prepare_keypoints(poses[frame])
        
        # Disegna connessioni
        for start_idx, end_idx in connections:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start = keypoints[start_idx]
                end = keypoints[end_idx]
                
                # Skip se punto non valido
                if not (abs(start[0]) < 1e-5 and abs(start[1]) < 1e-5) and \
                   not (abs(end[0]) < 1e-5 and abs(end[1]) < 1e-5):
                    line, = ax.plot([start[0], end[0]], [start[1], end[1]], 
                                   color='blue', linewidth=2.0, alpha=0.7)
                    lines.append(line)
        
        # Disegna keypoints
        valid_points_mask = ~np.all(np.abs(keypoints) < 1e-5, axis=1)
        valid_kp = keypoints[valid_points_mask]
        
        if len(valid_kp) > 0:
            points = ax.scatter(valid_kp[:, 0], valid_kp[:, 1], 
                              c='red', s=50, zorder=5, alpha=0.8)
        
        # Aggiorna testo
        token = tokens[frame]
        frame_text.set_text(f'Frame: {frame+1}/{n_frames}\nToken: {token}')
        
        return lines + [points, frame_text, caption_text]
    
    # Crea l'animazione
    interval = 1000 / fps  # Converti fps in millisecondi
    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        interval=interval, blit=True, repeat=True)
    
    # Salva se richiesto
    if save_path:
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"Animazione salvata in: {save_path}")
    
    # Mostra se richiesto
    if show:
        plt.tight_layout()
        plt.show()
    
    return anim


