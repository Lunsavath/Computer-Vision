#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:28:28 2019

@author: anush
"""

class Node(object):
    def __init__(self, data):
        self.data = data
        self.nextnode = None
        
class Linkedlist(object):
    def __init__(self):
        self.head = None
        self.size = 0
        
    def insertStart(self, data):
        self.size = self.size + 1
        newNode = Node(data)
        
        if not self.head:
            self.head = newNode
        else:
            newNode.nextnode = self.head
            self.head = newNode
            
    def remove(self,data):
        if self.head is None:
            return
        self.size = self.size - 1
        currentNode = self.head
        previousNode = None
        
        while currentNode.data != data:
            previousNode = currentNode
            currentNode = currentNode.nextnode
        
        if currentNode is None:
            self.head = currentNode.nextNode
        else:
            previousNode.nextnode = currentNode.nextnode
        
    
    def size1(self):
        return self.size
    
    def size2(self):
        actualNode = self.head
        size = 0
        while actualNode is not None:
            size += 1
            actualNode = actualNode.nextnode
        return size
    
    def insertEnd(self, data):
        self.size = self.size + 1
        newNode = Node(data)
        actualNode = self.head
        
        while actualNode.nextnode is not None:
            actualNode = actualNode.nextnode
        actualNode.nextnode = newNode
        
    def Traverselist(self):
        actualNode = self.head
        while actualNode is not None:
            print("%d " %actualNode.data)
            actualNode = actualNode.nextnode

linkedlist = Linkedlist()
linkedlist.insertStart(12)
linkedlist.insertStart(4)
linkedlist.insertStart(10)
linkedlist.insertStart(14)
linkedlist.insertEnd(7)

linkedlist.Traverselist()
linkedlist.size1()