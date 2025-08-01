# setup.py

import pathlib

from setuptools import find_packages, setup


def fetch_requirements():
    with open("requirements.txt") as f:
        reqs = f.read().strip().split("\n")
    return reqs


here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="dax",
    version="0.1.0",
    author="zhexin.lzx, yixuan.zyx, xingyuanjie.xyj, leyi.wjc",
    description="Diffusion Accelerated eXecution",
    long_description=long_description,  # 添加长描述
    long_description_content_type="text/markdown",  # 指定描述内容类型
    url="https://github.com/RiseAI-Sys/DAX.git",  # 主页面链接
    packages=find_packages(),
    install_requires=fetch_requirements(),
    entry_points={
        "console_scripts": [
            "my_project = my_project.main:main",
        ],
    },
    extras_require={},
    python_requires=">=3.10",
)
