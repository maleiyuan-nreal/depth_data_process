import os


def mergeFiles(obj, fileList: list) -> str:
    """

    :param fileList: a list of file absolute path
    :return: a string of merged file absolute path
    """
    fs = [open(file_, 'r') for file_ in fileList]
    tempDict = {}
    mergedFile = open(obj.NDS_FILE_NAME, 'a')
    for f in fs:
        initLine = f.readline()
        if initLine:
            tempDict[f] = initLine
    while tempDict:
        min_item = min(tempDict.items(), key=lambda x: x[1])
        mergedFile.write(min_item[1])
        nextLine = min_item[0].readline()
        if nextLine:
            tempDict[min_item[0]] = nextLine
        else:
            del tempDict[min_item[0]]
            min_item[0].close()
    mergedFile.close()
