# Tugas Reinforcement Learning — Implementasi Q-Learning pada FrozenLake menggunakan Python
# Andi Hani Tuzlimatul Izzati/202412013
# Matkul Kecerdasan Komputasional

## Tujuan
1. Memahami konsep Agent, State, Action, dan Reward.
2. Menerapkan algoritma Q-Learning.
3. Mengamati proses pembelajaran agent dalam mencari jalur terbaik.

---

## Langkah 1: Install Library
```bash
pip install gymnasium
```

## Langkah 2: Import Library
```python
import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
```

## Langkah 3: Membuat Environment
```python
env = gym.make("FrozenLake-v1", is_slippery=False)

print("Jumlah State :", env.observation_space.n)
print("Jumlah Action :", env.action_space.n)
```
**Output:**
```
Jumlah State : 16
Jumlah Action : 4
```

## Langkah 4: Inisialisasi Q-Table
```python
state_size = env.observation_space.n
action_size = env.action_space.n

q_table = np.zeros((state_size, action_size))

print(q_table)
```

## Langkah 5: Parameter Q-Learning
Sesuai konsep Q-Learning pada PPT yang menggunakan Learning Rate (α) dan Discount Factor (γ)
```python
alpha = 0.8       # learning rate
gamma = 0.95      # discount factor
epsilon = 1.0     # exploration rate

epsilon_decay = 0.995
min_epsilon = 0.01

episodes = 2000
max_steps = 100
```

## Langkah 6: Training Q-Learning
```python
rewards = []

for episode in range(episodes):

    state, info = env.reset()
    total_reward = 0

    for step in range(max_steps):

        # Exploration vs Exploitation
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, info = env.step(action)

        # Update Q-Table
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    rewards.append(total_reward)

print("Training selesai!")
```

## Langkah 7: Menampilkan Q-Table
```python
print("Q-Table Hasil Training")
print(q_table)
```
**Hasil Q-Table setelah training:**
```
[[0.735 0.774 0.774 0.735]
 [0.735 0.    0.815 0.769]
 [0.698 0.857 0.403 0.743]
 [0.721 0.    0.    0.   ]
 [0.774 0.815 0.    0.735]
 [0.    0.    0.    0.   ]
 [0.    0.903 0.    0.810]
 [0.    0.    0.    0.   ]
 [0.815 0.    0.857 0.774]
 [0.815 0.903 0.903 0.   ]
 [0.857 0.95  0.    0.857]
 [0.    0.    0.    0.   ]
 [0.    0.    0.    0.   ]
 [0.    0.903 0.95  0.857]
 [0.903 0.95  1.    0.903]
 [0.    0.    0.    0.   ]]
```

## Langkah 8: Visualisasi Reward
```python
plt.figure(figsize=(10,5))
plt.plot(rewards)
plt.title("Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.show()
```
Grafik menunjukkan reward yang awalnya rendah (banyak 0/gagal) lalu naik dan stabil di nilai 1 (berhasil mencapai goal) setelah agent belajar.

## Langkah 9: Testing Agent
```python
env = gym.make("FrozenLake-v1", render_mode="ansi", is_slippery=False)

state, info = env.reset()

done = False
total_reward = 0

while not done:

    action = np.argmax(q_table[state])

    state, reward, terminated, truncated, info = env.step(action)

    print(env.render())

    total_reward += reward

    done = terminated or truncated

print("Total Reward:", total_reward)
```
**Output:** Agent berhasil sampai goal dengan **Total Reward: 1.0**

---

## Jawaban Pertanyaan

**1. Apa yang dimaksud dengan State, Action, dan Reward dalam Reinforcement Learning?**
- **State (S)**: kondisi/posisi agent saat ini di environment. Pada FrozenLake, state adalah posisi agent di grid (0–15).
- **Action (A)**: pilihan gerakan yang bisa dilakukan agent pada suatu state (kiri, kanan, atas, bawah).
- **Reward (R)**: sinyal umpan balik berupa angka dari environment setelah agent melakukan action, menunjukkan seberapa baik/buruk action tersebut (misal +1 jika sampai goal, 0 jika belum, atau negatif jika jatuh ke lubang).

**2. Apa fungsi dari Learning Rate (α)?**
Mengatur seberapa besar update nilai Q-table setiap kali agent mendapat pengalaman baru. α besar → agent cepat menyesuaikan dengan info baru; α kecil → update lebih lambat tapi stabil.

**3. Apa fungsi dari Discount Factor (γ)?**
Untuk menentukan seberapa besar agent mempertimbangkan reward di masa depan dibanding reward sekarang. γ mendekati 1 agent "berpikir jangka panjang" γ mendekati 0 agent fokus reward langsung.

**4. Mengapa digunakan metode Exploration dan Exploitation?**
- **Exploration**: agent mencoba action baru yang belum diketahui hasilnya, agar bisa menemukan strategi lebih baik.
- **Exploitation**: agent memanfaatkan pengetahuan (nilai Q terbaik) yang sudah dipelajari untuk reward maksimal.

**5. Bagaimana perubahan nilai reward setelah training 2000 episode?**
Reward rata-rata 100 episode awal sangat rendah (~0.04) karena agent masih banyak gagal. Setelah 2000 episode, reward rata-rata 100 episode terakhir naik signifikan jadi ~0.98, artinya agent hampir selalu berhasil mencapai goal. Grafik reward per episode juga menunjukkan tren naik dan stabil di nilai tinggi.

---

## Tugas Lanjutan: Q-Learning pada Environment Taxi-v3

Modifikasi program agar:
1. Menggunakan environment **Taxi-v3**.
2. Menampilkan rata-rata reward setiap 100 episode.
3. Membandingkan hasil training 1000, 2000, dan 5000 episode.

```python
def train_taxi_agent(episodes, max_steps=200, alpha=0.8, gamma=0.95,
                      epsilon_start=1.0, epsilon_decay=0.995, min_epsilon=0.01):
    """Melatih agent Q-Learning pada environment Taxi-v3."""

    env = gym.make("Taxi-v3")

    state_size = env.observation_space.n
    action_size = env.action_space.n
    q_table = np.zeros((state_size, action_size))

    epsilon = epsilon_start
    rewards = []

    for episode in range(episodes):
        state, info = env.reset()
        total_reward = 0

        for step in range(max_steps):

            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, info = env.step(action)

            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards.append(total_reward)

    env.close()
    return q_table, rewards


def rata_rata_per_100_episode(rewards):
    """Menghitung rata-rata reward setiap 100 episode."""
    hasil = []
    for i in range(0, len(rewards), 100):
        rata = np.mean(rewards[i:i+100])
        hasil.append(rata)
    return hasil
```

```python
# Training dengan 1000, 2000, dan 5000 episode untuk dibandingkan
daftar_episode = [1000, 2000, 5000]
hasil_training = {}

for jumlah_episode in daftar_episode:
    print(f"Training Taxi-v3 dengan {jumlah_episode} episode...")
    q_table_taxi, rewards_taxi = train_taxi_agent(jumlah_episode)
    hasil_training[jumlah_episode] = {
        "q_table": q_table_taxi,
        "rewards": rewards_taxi,
        "avg_per_100": rata_rata_per_100_episode(rewards_taxi)
    }
    print(f"Selesai. Rata-rata reward 100 episode terakhir: {np.mean(rewards_taxi[-100:]):.2f}")
    print()
```

```python
# Visualisasi perbandingan rata-rata reward per 100 episode
plt.figure(figsize=(10,6))

for jumlah_episode in daftar_episode:
    avg_per_100 = hasil_training[jumlah_episode]["avg_per_100"]
    x = [(i+1)*100 for i in range(len(avg_per_100))]
    plt.plot(x, avg_per_100, label=f"{jumlah_episode} episode")

plt.title("Perbandingan Rata-rata Reward per 100 Episode (Taxi-v3)")
plt.xlabel("Episode")
plt.ylabel("Rata-rata Reward (per 100 episode)")
plt.legend()
plt.grid(True)
plt.show()
```

```python
# Tabel perbandingan ringkas hasil training
print("Perbandingan Hasil Training Taxi-v3")
print("-" * 50)
for jumlah_episode in daftar_episode:
    avg_terakhir = np.mean(hasil_training[jumlah_episode]["rewards"][-100:])
    print(f"Episode = {jumlah_episode:5d} | Rata-rata reward 100 episode terakhir = {avg_terakhir:.2f}")
```

**Contoh hasil perbandingan**

| Jumlah Episode | Rata-rata Reward (100 episode terakhir) |
|---|---|
| 1000 | 7.32 |
| 2000 | 7.31 |
| 5000 | 7.73 |

### Analisis
Dari hasil training Taxi-v3 dengan 1000, 2000, dan 5000 episode terlihat bahwa:
- Pada awal training, reward sangat negatif karena agent banyak melakukan kesalahan (aksi tidak valid atau butuh waktu lama mengantar penumpang).
- Semakin banyak episode training, rata-rata reward per 100 episode semakin meningkat dan stabil di nilai positif — tanda agent sudah menemukan strategi mendekati optimal.
- Training dengan 5000 episode umumnya menghasilkan kebijakan (policy) yang lebih stabil dibanding 1000 episode, karena agent punya lebih banyak kesempatan mengeksplorasi state space yang jauh lebih besar dari FrozenLake.
