import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import product
from enum import Enum, auto
from dataclasses import dataclass


class NodeKind(Enum):
    NORMAL = auto()
    TSV = auto()
    MEMCTRL = auto()


@dataclass
class NodeType:
    kind: NodeKind = None
    chiplet_no: int = None


class HopSim:
    def __init__(self, argConfig):

        self.config = argConfig
        self.icn = nx.Graph()

        self.numXDimNodes = int(self.config['topology']['numxdimnodes'])
        self.numYDimNodes = int(self.config['topology']['numydimnodes'])
        self.numTotalNodes = self.numXDimNodes * self.numYDimNodes
        self.numTotalMemCtrl = 2 * self.numXDimNodes

        self.topolgyList = self.config['topology']['type'].split(' ')
        self.isSquareList = list(map(int, self.config['topology']['isSquare'].split(' ')))
        self.tsvPatternTypeList = self.config['tsv']['tsvpatterntype'].split(' ')

        self.topolgy = None
        self.isSquare = None
        self.tsvLayout: tuple = None

        self.tsvIndexList = []

        self.numTotalTSV = None

        self.probCoreToCore = None
        self.probCoreToMemCtrl = None
        self.probMemCtrlToCore = None

        self.tsvDispListSquare = None
        self.tsvDispListNotSquare = None

    def __get2DIndex(self, arg1DIndex):
        return (arg1DIndex % self.numXDimNodes, arg1DIndex // self.numXDimNodes)

    def __get1DIndex(self, arg2DIndex: tuple):
        (indexX, indexY) = arg2DIndex

        return indexY * self.numXDimNodes + indexX
    
    def __place(self) -> tuple:

        tsvDispDict = {}

        if(self.isSquare):
            chipletOffsetList = [8, 12, 64, 68]
        else:
            chipletOffsetList = [8, 32, 72, 96]

        for chipletNo in range(len(self.tsvLayout)):
            tsvPattern = self.tsvLayout[chipletNo]

            match tsvPattern:
                case 'border':
                    if(self.isSquare):
                        tsvDispList = [0, 1, 2, 3, 8, 11, 16, 19, \
                                        24, 27, 32, 35, 40, 43, 48, 49, 50, 51]
                    else:
                        tsvDispList = [0, 1, 2, 3, 4, 5, 6, 7, 
                                    8, 15, 
                                    16, 17, 18, 19, 20, 21, 22, 23]
                        
                case 'bundle':
                    if(self.isSquare):
                        tsvDispList = [9, 10, 17, 18, 25, 26, 33, 34, 41, 42]
                    else:
                        tsvDispList = [9, 10, 11, 12, 13, 14]

                case 'shielded':
                    if(self.isSquare):
                        tsvDispList = [9, 10, 25, 26, 41, 42]
                    else:
                        tsvDispList = [9, 10, 13, 14]

                case 'isolated':
                    tsvDispList = []

                    if(self.isSquare):
                        if(None != self.tsvDispListSquare):
                            tsvDispList = self.tsvDispListSquare
                        else:
                            maxTSVDisp = 28
                            numTSVPerChiplet = 7

                            while(len(tsvDispList) < numTSVPerChiplet):
                                tsvDisp = random.randint(0, maxTSVDisp)

                                if(tsvDisp not in tsvDispList):
                                    tsvDispList.append(tsvDisp)

                            self.tsvDispListSquare = tsvDispList
                    else:
                        if(None != self.tsvDispListNotSquare):
                            tsvDispList = self.tsvDispListNotSquare
                        else:
                            maxTSVDisp = 24
                            numTSVPerChiplet = 6

                            while(len(tsvDispList) < numTSVPerChiplet):
                                tsvDisp = random.randint(0, maxTSVDisp)

                                if(tsvDisp not in tsvDispList):
                                    tsvDispList.append(tsvDisp)

                            self.tsvDispListNotSquare = tsvDispList

                case _:
                    chipletOffsetList = []
                    tsvDispList = []

            tsvDispDict[chipletNo] = tsvDispList

        self.tsvIndexList.clear()

        for chipletNo in range(len(chipletOffsetList)):
            for tsvDisp in tsvDispDict[chipletNo]:
                self.tsvIndexList.append(self.__get2DIndex(chipletOffsetList[chipletNo] + tsvDisp))

        self.numTotalTSV = len(tsvDispDict[0]) + len(tsvDispDict[1]) +\
        len(tsvDispDict[2]) + len(tsvDispDict[3])

        self.probCoreToCore = np.float32(0.3 / self.numTotalTSV)
        self.probCoreToMemCtrl = np.float32(0.7 / self.numTotalMemCtrl)
        self.probMemCtrlToCore = np.float32(1/self.numTotalTSV)

    def __setTopolgy(self):
        for nodeId in range(self.numTotalNodes):
            self.icn.add_node(nodeId)

        match self.topolgy:
            case 'mesh':
                self.__setMesh()

            case 'cmesh':
                self.__setCMesh()

            case 'dbutterfly':
                self.__setDButterfly()

            case 'ftorus':
                self.__setFTorus()

            case 'bdonut':
                self.__setBDonut()

            case _:
                pass

        self.__setNodeType()

    def __getChipletNo(self, argIndexX, argIndexY):
        if(self.isSquare):  #grid
            if(argIndexY in [0, self.numYDimNodes]):
                return -1   # does not belong to a chiplet
            else:
                return (argIndexY // 8) * 2 + (argIndexX // 4)
        else:   #list
            if(argIndexY in [0, 7, 8, self.numYDimNodes]):
                return -1   # does not belong to a chiplet
            else:
                return argIndexY // 4

    def __setNodeType(self):
        for (srcIndexX, srcIndexY) in product(range(self.numXDimNodes), range(self.numYDimNodes)):
            # Chiplet number that this node belongs to
            chipletNo = self.__getChipletNo(srcIndexX, srcIndexY)

            src1DIndex = self.__get1DIndex((srcIndexX, srcIndexY))
            nodeKind = NodeKind.NORMAL

            if (srcIndexY in [0, self.numYDimNodes]):
                # Edge case
                nodeKind = NodeKind.MEMCTRL

            elif ((srcIndexX, srcIndexY) in self.tsvIndexList):
                nodeKind = NodeKind.TSV

            self.icn.nodes[src1DIndex]['type'] = NodeType(
                kind=nodeKind, chiplet_no=chipletNo)

    def __setMesh(self):
        for (srcIndexX, srcIndexY) in product(range(self.numXDimNodes), range(self.numYDimNodes)):
            dstIndexX = srcIndexX
            dstIndexY = srcIndexY + 1

            if (self.numYDimNodes <= dstIndexY):
                continue

            srcNodeId = self.__get1DIndex((srcIndexX, srcIndexY))
            dstNodeId = self.__get1DIndex((dstIndexX, dstIndexY))

            self.icn.add_edge(srcNodeId, dstNodeId)
            self.icn.add_edge(dstNodeId, srcNodeId)

        for (srcIndexY, srcIndexX) in product(range(self.numYDimNodes), range(self.numXDimNodes)):
            dstIndexX = srcIndexX + 1
            dstindexY = srcIndexY

            if (self.numXDimNodes <= dstIndexX):
                continue

            srcNodeId = self.__get1DIndex((srcIndexX, srcIndexY))
            dstNodeId = self.__get1DIndex((dstIndexX, dstindexY))

            self.icn.add_edge(srcNodeId, dstNodeId)
            self.icn.add_edge(dstNodeId, srcNodeId)

        return 1

    def __setCMesh(self):

        pass

    def __setDButterfly(self):
        # Horizontal links
        for (srcIndexY, srcIndexX) in product(range(self.numYDimNodes), range(self.numXDimNodes)):
            dstIndexX = srcIndexX + 1
            dstIndexY = srcIndexY

            if (self.numXDimNodes <= dstIndexX):
                continue

            srcNodeId = self.__get1DIndex((srcIndexX, srcIndexY))
            dstNodeId = self.__get1DIndex((dstIndexX, dstIndexY))

            self.icn.add_edge(srcNodeId, dstNodeId)
            self.icn.add_edge(dstNodeId, srcNodeId)

        # Vertical links
        for (srcIndexX, srcIndexY) in product(range(self.numXDimNodes - 1), range(self.numYDimNodes)):
            dstIndexX = srcIndexX + 1

            if srcIndexX == 0 or srcIndexX == 6:
                dstIndexX = srcIndexX + 1
                dstIndexY = (srcIndexY // 2) * 2 + ((srcIndexY + 1) % 2)
                srcNodeId = self.__get1DIndex((srcIndexX, srcIndexY))
                dstNodeId = self.__get1DIndex((dstIndexX, dstIndexY))

                self.icn.add_edge(srcNodeId, dstNodeId)
                self.icn.add_edge(dstNodeId, srcNodeId)

            elif srcIndexX == 1 or srcIndexX == 5:
                dstIndexX = srcIndexX + 1
                dstIndexY = (srcIndexY // 4) * 4 + ((srcIndexY + 2) % 4)
                srcNodeId = self.__get1DIndex((srcIndexX, srcIndexY))
                dstNodeId = self.__get1DIndex((dstIndexX, dstIndexY))

                self.icn.add_edge(srcNodeId, dstNodeId)
                self.icn.add_edge(dstNodeId, srcNodeId)

            elif srcIndexX == 2 or srcIndexX == 4:
                dstIndexX = srcIndexX + 1
                dstIndexY = (srcIndexY // 8) * 8 + ((srcIndexY + 4) % 8)
                srcNodeId = self.__get1DIndex((srcIndexX, srcIndexY))
                dstNodeId = self.__get1DIndex((dstIndexX, dstIndexY))

                self.icn.add_edge(srcNodeId, dstNodeId)
                self.icn.add_edge(dstNodeId, srcNodeId)

            elif srcIndexX == 3:
                dstIndexX = srcIndexX + 1
                dstIndexY = (srcIndexY // 16) * 16 + ((srcIndexY + 8) % 16)
                srcNodeId = self.__get1DIndex((srcIndexX, srcIndexY))
                dstNodeId = self.__get1DIndex((dstIndexX, dstIndexY))

                self.icn.add_edge(srcNodeId, dstNodeId)
                self.icn.add_edge(dstNodeId, srcNodeId)

        return

    def __setFTorus(self):
        # Horizontal links
        for (srcIndexY, srcIndexX) in product(range(self.numYDimNodes), range(self.numXDimNodes)):
            dstIndexY = srcIndexY

            xdst_lst = []
            if srcIndexX == self.numXDimNodes - 2:
                xdst_lst.append(self.numXDimNodes - 1)
            else:
                xdst_lst.append(srcIndexX + 2)
                if srcIndexX == 0:
                    xdst_lst.append(1)

            for xdst in xdst_lst:
                if (self.numXDimNodes <= xdst):
                    continue

                srcNodeId = self.__get1DIndex((srcIndexX, srcIndexY))
                dstNodeId = self.__get1DIndex((xdst, dstIndexY))

                self.icn.add_edge(srcNodeId, dstNodeId)
                self.icn.add_edge(dstNodeId, srcNodeId)

        # Vertical links
        for (srcIndexY, srcIndexX) in product(range(self.numYDimNodes), range(self.numXDimNodes)):
            dstIndexX = srcIndexX

            ydst_lst = []
            if srcIndexY == self.numYDimNodes - 2:
                ydst_lst.append(self.numYDimNodes - 1)
            else:
                ydst_lst.append(srcIndexY + 2)
                if srcIndexY == 0:
                    ydst_lst.append(1)

            for ydst in ydst_lst:
                if (self.numYDimNodes <= ydst):
                    continue

                srcNodeId = self.__get1DIndex((srcIndexX, srcIndexY))
                dstNodeId = self.__get1DIndex((dstIndexX, ydst))

                self.icn.add_edge(srcNodeId, dstNodeId)
                self.icn.add_edge(dstNodeId, srcNodeId)

        return

    def __setBDonut(self):
        # Horizontal links
        for (srcIndexY, srcIndexX) in product(range(self.numYDimNodes), range(self.numXDimNodes)):
            dstIndexY = srcIndexY

            xdst_lst = []
            if srcIndexX == self.numXDimNodes - 2:
                xdst_lst.append(self.numXDimNodes - 1)
            else:
                xdst_lst.append(srcIndexX + 2)
                if srcIndexX == 0:
                    xdst_lst.append(1)

            for xdst in xdst_lst:
                if (self.numXDimNodes <= xdst):
                    continue

                srcNodeId = self.__get1DIndex((srcIndexX, srcIndexY))
                dstNodeId = self.__get1DIndex((xdst, dstIndexY))

                self.icn.add_edge(srcNodeId, dstNodeId)
                self.icn.add_edge(dstNodeId, srcNodeId)

        # Vertical links
        for (srcIndexX, srcIndexY) in product(range(self.numXDimNodes - 1), range(self.numYDimNodes)):
            dstIndexX = srcIndexX + 1

            if srcIndexX == 0 or srcIndexX == 6:
                dstIndexX = srcIndexX + 1
                dstIndexY = (srcIndexY // 2) * 2 + ((srcIndexY + 1) % 2)
                srcNodeId = self.__get1DIndex((srcIndexX, srcIndexY))
                dstNodeId = self.__get1DIndex((dstIndexX, dstIndexY))

                self.icn.add_edge(srcNodeId, dstNodeId)
                self.icn.add_edge(dstNodeId, srcNodeId)

            elif srcIndexX == 1 or srcIndexX == 5:
                dstIndexX = srcIndexX + 1
                dstIndexY = (srcIndexY // 4) * 4 + ((srcIndexY + 2) % 4)
                srcNodeId = self.__get1DIndex((srcIndexX, srcIndexY))
                dstNodeId = self.__get1DIndex((dstIndexX, dstIndexY))

                self.icn.add_edge(srcNodeId, dstNodeId)
                self.icn.add_edge(dstNodeId, srcNodeId)

            elif srcIndexX == 2 or srcIndexX == 4:
                dstIndexX = srcIndexX + 1
                dstIndexY = (srcIndexY // 8) * 8 + ((srcIndexY + 4) % 8)
                srcNodeId = self.__get1DIndex((srcIndexX, srcIndexY))
                dstNodeId = self.__get1DIndex((dstIndexX, dstIndexY))

                self.icn.add_edge(srcNodeId, dstNodeId)
                self.icn.add_edge(dstNodeId, srcNodeId)

            elif srcIndexX == 3:
                dstIndexX = srcIndexX + 1
                dstIndexY = (srcIndexY // 16) * 16 + ((srcIndexY + 8) % 16)
                srcNodeId = self.__get1DIndex((srcIndexX, srcIndexY))
                dstNodeId = self.__get1DIndex((dstIndexX, dstIndexY))

                self.icn.add_edge(srcNodeId, dstNodeId)
                self.icn.add_edge(dstNodeId, srcNodeId)

        return

    def __visualize(self):
        # dict with two of the positions set
        fixed_positions = {}
        for (y, x) in product(range(self.numYDimNodes), range(self.numXDimNodes)):
            node_num = x + (self.numXDimNodes * (self.numYDimNodes - 1 - y))
            fixed_positions[node_num] = (x, y)

        fixed_nodes = fixed_positions.keys()

        pos = nx.spring_layout(
            self.icn, dim=2, pos=fixed_positions, fixed=fixed_nodes)

        nx.draw(self.icn, pos, with_labels=True,
                node_color='lightblue', edge_color='gray', node_size = 200)

        figName = self.topolgy + ".png"

        plt.savefig(figName)
        plt.cla()

    def visualize(self):
        for topology in self.topolgyList:
            self.topolgy = topology

            self.__clear()
            self.__setTopolgy()
            self.__visualize()


    def __getNodeType(self, argNodeId: int) -> NodeType:
        return self.icn.nodes[argNodeId]['type']

    def __getNodeKind(self, argNodeId: int) -> NodeKind:
        return self.__getNodeType(argNodeId).kind

    def __getNodeChipletNo(self, argNodeId: int) -> int:
        return self.__getNodeType(argNodeId).chiplet_no

    def __getHopCountBetween(self, argSrcNodeId: int, argDstNodeId: int) -> int:
        return (len(nx.shortest_path(self.icn, argSrcNodeId, argDstNodeId)) - 1)

    def __getAvgHopCountAt(self, argNodeId: int) -> np.float32:
        match self.__getNodeKind(argNodeId):
            case NodeKind.NORMAL:
                return 0

            case NodeKind.TSV:
                avgHopCount = 0
                for dstNodeId in range(self.numTotalNodes):
                    match self.__getNodeKind(dstNodeId):
                        case NodeKind.NORMAL:
                            pass
                        case NodeKind.TSV if int(self.__getNodeChipletNo(argNodeId)) != int(self.__getNodeChipletNo(dstNodeId)):
                            avgHopCount += self.probCoreToCore * \
                                self.__getHopCountBetween(
                                    argNodeId, dstNodeId)

                        case NodeKind.MEMCTRL:
                            avgHopCount += self.probCoreToMemCtrl * \
                                self.__getHopCountBetween(argNodeId, dstNodeId)

                        case _:
                            pass

                return np.float32(avgHopCount)

            case NodeKind.MEMCTRL:
                avgHopCount = 0

                for dstNodeId in range(self.numTotalNodes):
                    match self.__getNodeType(dstNodeId).kind:
                        case NodeKind.NORMAL:
                            pass
                        case NodeKind.TSV:
                            avgHopCount += self.probMemCtrlToCore * \
                                self.__getHopCountBetween(argNodeId, dstNodeId)

                        case NodeKind.MEMCTRL:
                            pass

                        case _:
                            pass

                return np.float32(avgHopCount)

            case _:
                pass

    def __clear(self):
        return self.icn.clear()

    def __getPossibleTSVLayout(self):
        tsvPatternTypeList = self.tsvPatternTypeList
        tsvLayoutList = []

        for chiplet1TSVType in tsvPatternTypeList:
            for chiplet2TSVType in tsvPatternTypeList:
                for chiplet3TSVType in tsvPatternTypeList:
                    for chiplet4TSVType in tsvPatternTypeList:
                        tsvLayout = (chiplet1TSVType, chiplet2TSVType,\
                                     chiplet3TSVType, chiplet4TSVType)
                        tsvLayoutList.append(tsvLayout)

        return tsvLayoutList
                        
    def __run(self):
        # returns avg hop count

        hopCountSum = 0

        for indexX in range(self.numXDimNodes):
            for indexY in range(self.numYDimNodes):
                targetNodeId = self.__get1DIndex((indexX, indexY))
                hopCountSum += self.__getAvgHopCountAt(targetNodeId)

        return np.float32(hopCountSum/(self.numTotalTSV + self.numTotalMemCtrl))
    
    def run(self):

        topologyIndexList = []
        meanList = []
        tsvLayoutList = self.__getPossibleTSVLayout()

        plt.ylim(0, 8)

        for topology in self.topolgyList:
            print(f"======= Topology: {topology}")

            self.topolgy = topology
            hopCountList = []

            for isSquare in self.isSquareList:
                if(isSquare):
                    isSquareStr = 'Grid'
                else:
                    isSquareStr = 'List'

                for tsvLayout in tsvLayoutList:
                    (self.isSquare, self.tsvLayout) = isSquare, tsvLayout

                    self.__clear()
                    self.__place()
                    self.__setTopolgy()
                    avgHopCount = self.__run()
                    hopCountList.append(avgHopCount)

                    #print(f"HopCount when {tsvLayout}, {isSquareStr}: {avgHopCount}")

            stdDev = np.std(hopCountList)
            mean = np.mean(hopCountList)
            meanList.append(mean)

            print(f"StdDev: {stdDev}")
            print(f"Mean: {mean}")
            print()

            topologyIndex = self.topolgyList.index(topology)
            topologyIndexList.append(topologyIndex)

            for hopCount in hopCountList:
                plt.scatter(topologyIndex, hopCount, s = 30, color = 'blue')



        plt.plot(topologyIndexList, meanList, marker = 's', linestyle = '--', color = 'orange')

        plt.xticks(topologyIndexList, self.topolgyList)

        plt.xlabel("NoI Topology")
        plt.ylabel("Averge Hop Count")
        plt.title("Average Hop Counts Of Different NoI Topologies")

        plt.savefig("result.png")
        plt.cla()



    # for debugging purposes
    def checkNodeType(self):
        for index in range(self.numTotalNodes):
            print(self.__getNodeKind(index).name,
                  self.__get2DIndex(index))
