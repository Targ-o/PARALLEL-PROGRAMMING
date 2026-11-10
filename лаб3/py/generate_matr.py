import random

sizes = [20, 200, 400, 800, 1200, 1600, 2000]

random.seed(42)

for n in sizes:
    for name in ["a", "b"]:
        filename = f"data/matrix_{name}_{n}.txt"

        with open(filename, "w") as f:
            f.write(f"{n}\n")

            for i in range(n):
                row = [str(random.randint(1, 10)) for _ in range(n)]
                f.write(" ".join(row) + "\n")

        print(f"Created {filename}")

print("All matrices generated!")
