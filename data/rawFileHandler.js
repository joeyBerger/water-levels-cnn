const { promisify } = require('util');
const fs = require('fs');
const convert = require('heic-convert');
const im = require('imagemagick');

const rawImagePath = 'rawImages';
const convertedImagePath = 'convertedImages';
const rawFileType = 'HEIC';
const convertedFileType = 'jpg';

const pathIsDir = (path) => {
  try {
    const stats = fs.statSync(path);
    return stats.isDirectory();
  } catch (error) {
    console.error(`Error checking path: ${error.message}`);
    return false;
  }
};

const createPath = (path, paths) => {
  path += paths.shift() + '/';
  if (!fs.existsSync(path)) fs.mkdirSync(path);
  if (paths.length) createPath(path, paths);
};

const deleteFolderRecursive = (path) => {
  if (fs.existsSync(path)) {
    fs.readdirSync(path).forEach((file, index) => {
      var curPath = path + '/' + file;
      if (fs.lstatSync(curPath).isDirectory()) {
        deleteFolderRecursive(curPath);
      } else {
        fs.unlinkSync(curPath);
      }
    });
    fs.rmdirSync(path);
  }
};

const readDirs = (path) => {
  let dirs = fs.readdirSync(path);
  return dirs.filter((p) => {
    pathIsDir(`${path}/${p}`);
    return pathIsDir(`${path}/${p}`);
  });
};

const convertFiles = async () => {
  let dirs = readDirs(rawImagePath);

  dirs.forEach(async (dir) => {
    const files = fs.readdirSync(`${rawImagePath}/${dir}`).filter((file) => file.includes(rawFileType));

    const fullDestinationPath = `${convertedImagePath}/${dir}`;
    deleteFolderRecursive(fullDestinationPath);
    createPath('', fullDestinationPath.split('/'));

    const promises = files.map(async (file) => {
      const newFileName = file.replace(rawFileType, convertedFileType);
      const inputBuffer = await promisify(fs.readFile)(`${rawImagePath}/${dir}/${file}`);
      const outputBuffer = await convert({
        buffer: inputBuffer, // the HEIC file buffer
        format: 'JPEG', // output format
        quality: 1, // the jpeg compression quality, between 0 and 1
      });
      return promisify(fs.writeFile)(`${convertedImagePath}/${dir}/${newFileName}`, outputBuffer);
    });
    await Promise.all[promises];
  });
};

const resizeViaImageMagick = async () => {
  let dirs = readDirs(rawImagePath);

  dirs.forEach(async (dir) => {
    const files = fs.readdirSync(`${rawImagePath}/${dir}`).filter((file) => file.includes(rawFileType));

    const fullDestinationPath = `${convertedImagePath}/${dir}`;
    deleteFolderRecursive(fullDestinationPath);
    createPath('', fullDestinationPath.split('/'));

    files.forEach(async (file) => {
      const newFileName = file.replace(rawFileType, convertedFileType);
      im.resize(
        {
          srcPath: `${rawImagePath}/${dir}/${file}`,
          dstPath: `${convertedImagePath}/${dir}/${newFileName}`,
          width: 500,
        },
        (err, stdout, stderr) => {
          if (err) throw err;
          console.log(`Converted file: ${file} to ${newFileName}`);
        }
      );
    });
  });
};

const separateImagesForModel = (splitInfo) => {
  let dirs = readDirs(convertedImagePath);
  Object.keys(splitInfo).forEach((k, _, arr) => {
    deleteFolderRecursive(k);
    createPath('', [k, '']);
    dirs.forEach((dir) => {
      createPath('', [k, dir]);
      const allFiles = fs
        .readdirSync(`${convertedImagePath}/${dir}`)
        .filter((file) => file.includes(convertedFileType));

      const filesToSelect = Math.floor(allFiles.length * splitInfo[k].share);

      for (let i = 0; i < filesToSelect; ) {
        const idx = Math.floor(Math.random() * allFiles.length);
        const fileExists = arr.find((k) => {
          const fileNames = splitInfo[k].selectedFiles.map((info) => info.fileName);
          return fileNames.includes(allFiles[idx]);
        });

        if (!fileExists) {
          splitInfo[k].selectedFiles.push({
            fileName: allFiles[idx],
            folderName: dir,
          });
          i++;
        }
      }
    });

    // dirs.forEach((dir) => {
    //   const allFiles = fs
    //     .readdirSync(`${convertedImagePath}/${dir}`)
    //     .filter((file) => file.includes(convertedFileType));

    //   const usedDirFiles = Object.keys(splitInfo)
    //     .map((k) => splitInfo[k].selectedFiles)
    //     .flat()
    //     .filter((info) => info.folderName === dir)
    //     .map((info) => info.fileName);

    //   const leftoverFiles = allFiles.filter((file) => !usedDirFiles.includes(file));

    //   leftoverFiles.forEach((file) => {
    //     const idx = Math.floor(Math.random() * arr.length);
    //     splitInfo[arr[idx]].selectedFiles.push({
    //       fileName: file,
    //       folderName: dir,
    //     });
    //   });

    // });
  });

  Object.keys(splitInfo).forEach((k, _, arr) => {
    splitInfo[k].selectedFiles
      .filter((info) => info.folderName !== '')
      .forEach((info) => {
        fs.copyFileSync(
          `${convertedImagePath}/${info.folderName}/${info.fileName}`,
          `${k}/${info.folderName}/${info.fileName}`
        );
      });
  });
};

//Get meta data example
/*
im.readMetadata(`${rawImagePath}/1dl/IMG_1113.HEIC`, (err, metadata) => {
  if (err) throw err;
  console.log(
    Object.keys(metadata.exif)
      .map((obj) => obj)
      .join('\n')
  );
  console.log(metadata.exif.xResolution);
  console.log(metadata.exif.pixelXDimension);
  console.log(metadata.exif.pixelYDimension);
});
*/

// convertFiles()
// resizeViaImageMagick();
separateImagesForModel({
  train: {
    share: 0.8,
    selectedFiles: [
      {
        fileName: '',
        folderName: '',
      },
    ],
  },
  valid: {
    share: 0.1,
    selectedFiles: [
      {
        fileName: '',
        folderName: '',
      },
    ],
  },
  test: {
    share: 0.1,
    selectedFiles: [
      {
        fileName: '',
        folderName: '',
      },
    ],
  },
});
