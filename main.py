import os
from process_image import process_image, other


def select_files(directory: str, max_files=10) -> list[str]:
    current_dir = os.getcwd()
    dir = f"{current_dir}/{directory}"
    all_files = os.listdir(dir)
    # for f in all_files:
    #     p = os.path.join(dir, f)
    #     print(p)

    # # (print(f) for f in all_files if os.path.isfile(os.path.join(dir, f)))
    # if all_files is None:
    #     return []

    files = [os.path.join(dir, f) for f in all_files]

    return files[:max_files]


if __name__ == "__main__":
    # other("Cards/CO-DAR")
    files = select_files("Cards/CO-DAR")
    for file in files:

        img_txt = other(file)
        with open("output1.txt", "a") as f:
            f.write(f"{img_txt}\n")
