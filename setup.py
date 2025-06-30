import setuptools

setuptools.setup(
    name="process_mixingtime",
    version="0.0.1",
    description="Tools to process images for mixing time analysis.",
    long_description="Tools to process images for mixing time. Using the GUI, you can select images and process them to analyze the mixing time when a dye is inserted into the fluid.",
    url="",
    author="Weheliye Hashi",
    author_email="Weheliye.Weheliye@oribiotech.com",
    license="MIT",
    packages=setuptools.find_packages(),
    zip_safe=False,
    entry_points={
         'console_scripts': [
            'mt_gui=process_mixingtime.mixingtime_gui:main',
            'mt_main_processing=process_mixingtime.Process_main_images_GUI:main',
        ],
    },
)