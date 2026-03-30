import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from train import train, ReachingEnv, _augment_state

def main():
    print("Entraînement du modèle en cours...")
    results = train(n_episodes=1000, print_every=500)
    env, cortex, bg, fwd, inv, _, _, _ = results

    # Initialisation d'un environnement de test
    np.random.seed(42)
    raw_state = env.reset()
    start_pos = env.pos.copy()
    target_pos = env.target.copy()
    
    # Tu peux changer cette variable pour voir la différence de mouvement :
    # 'bg' (Ganglions), 'inverse' (Cervelet), 'predictive' (Cervelet + Ganglions)
    strategy = 'bg' 
    print(f"\nGénération de l'animation avec la stratégie : {strategy.upper()}")
    
    trajectory = [start_pos.copy()]
    done = False
    
    # --- Capture de la trajectoire selon la stratégie ---
    if strategy == 'bg':
        cortex_repr = cortex.encode(raw_state)
        bg_state = _augment_state(raw_state, cortex_repr)
        while not done:
            action = bg.policy(bg_state, explore=False)
            raw_state, r, done = env.step(action)
            trajectory.append(env.pos.copy())
            cortex_repr = cortex.encode(raw_state)
            bg_state = _augment_state(raw_state, cortex_repr)
            
    elif strategy == 'inverse':
        target_raw = np.concatenate([target_pos, np.zeros(2), target_pos])
        while not done:
            action = inv.compute_action(raw_state, target_raw)
            action = np.clip(action, -1, 1)
            raw_state, r, done = env.step(action)
            trajectory.append(env.pos.copy())
            raw_state = env._get_state()
            
    elif strategy == 'predictive':
        cortex_repr = cortex.encode(raw_state)
        bg_state = _augment_state(raw_state, cortex_repr)
        n_candidates = 20
        while not done:
            base_action = bg.policy(bg_state, explore=False)
            best_action = base_action.copy()
            best_value = -1e10
            for _ in range(n_candidates):
                candidate = base_action + np.random.randn(2) * 0.3
                candidate = np.clip(candidate, -1, 1)
                pred_next_raw = fwd.predict_next_state(raw_state, candidate)
                pred_cortex = cortex.encode(pred_next_raw)
                pred_bg_state = _augment_state(pred_next_raw, pred_cortex)
                v = bg.value(pred_bg_state)
                if v > best_value:
                    best_value = v
                    best_action = candidate
            raw_state, r, done = env.step(best_action)
            trajectory.append(env.pos.copy())
            raw_state = env._get_state()
            cortex_repr = cortex.encode(raw_state)
            bg_state = _augment_state(raw_state, cortex_repr)

    trajectory = np.array(trajectory)

    # --- Configuration de l'animation ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title(f"Dynamique d'atteinte - Stratégie : {strategy.upper()}")
    ax.set_xlabel("Espace moteur X")
    ax.set_ylabel("Espace moteur Y")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Dessiner la cible et le point de départ
    target_circle = plt.Circle((target_pos[0], target_pos[1]), 0.15, color='r', fill=False, linestyle='--')
    ax.add_patch(target_circle)
    ax.plot(target_pos[0], target_pos[1], 'r*', markersize=12, label="Cible")
    ax.plot(start_pos[0], start_pos[1], 'ko', markersize=8, label="Départ")
    
    # Éléments animés
    line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.6, label="Trajectoire")
    point, = ax.plot([], [], 'bo', markersize=8, label="Effecteur")
    
    ax.legend(loc="upper left")

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def animate(i):
        # Mettre à jour la ligne (trajectoire passée)
        line.set_data(trajectory[:i+1, 0], trajectory[:i+1, 1])
        # Mettre à jour le point (position actuelle)
        point.set_data([trajectory[i, 0]], [trajectory[i, 1]])
        return line, point

    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(trajectory), interval=100, blit=True, repeat=False
    )
    
    print("\nAffichage de l'animation...")
    plt.show()

if __name__ == "__main__":
    main()