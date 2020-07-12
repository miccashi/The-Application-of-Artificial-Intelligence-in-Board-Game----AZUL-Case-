from collections import deque
from tkinter import *
from tkinter import font

WIDTH = 1600
HEIGHT = 900


class Gui:
    def __init__(self, tree, name = 'tk'):
        self.tree = tree
        master = Tk()
        master.title(name)
        self.master = master
        w, h = master.maxsize()
        master.geometry("{}x{}".format(w, h))

        self.rate = 0.01
        self.elements = []
        self.sketch = []
        self.map_x = WIDTH // 2
        self.map_y = HEIGHT // 2
        self.show_info = False

        self.initWedge()
        self.master.protocol('WM_DELETE_WINDOW', self.closeWindow)
        master.mainloop()

    def closeWindow(self):
        print('closing window')
        self.master.destroy()

    def initWedge(self):
        master = self.master
        self.cv = Canvas(master,
                         width=WIDTH,
                         height=HEIGHT)
        self.cv.pack()
        self.cv.bind('<MouseWheel>', self.zoom)
        self.cv.bind('<B1-Motion>', self.move_to_map)
        self.cv.bind('<Button-1>', self.press)
        self.cv.bind('<Button-3>',self.info_switch)
        self.refresh_tree()

    def info_switch(self, event):
        if self.show_info is False:
            self.refresh_tree(info=True)
            self.show_info = True
        else:
            self.refresh_tree(info=False)
            self.show_info = False

    def press(self, event):
        self.pre_event_x = event.x
        self.pre_event_y = event.y

    def move_to_map(self, event):
        for s in self.sketch:
            self.cv.move(s, event.x - self.pre_event_x, event.y - self.pre_event_y)
        self.pre_event_x = event.x
        self.pre_event_y = event.y

    def zoom(self, event):
        zoom_rate = 0.8
        if event.delta < 0:
            for s in self.sketch:
                self.cv.scale(s, event.x, event.y, zoom_rate, zoom_rate)
        else:
            for s in self.sketch:
                self.cv.scale(s, event.x, event.y, 1 / zoom_rate, 1 / zoom_rate)

    def refresh_tree(self, info=False):
        for e in self.elements:
            e.delete_self()
        m = Map(x=self.map_x, y=self.map_y, rate=self.rate, cv=self.cv, elements=self.elements, sketch=self.sketch)
        self.m = m
        self.generate_tree(m, info)

    def generate_gui_node(self, node, parent_gui_node, m, info=False):
        DEFAULT_WIDTH = 200
        DEFAULT_HEIGTH = 200
        DEFAULT_LAYERH_HEIGHT = 2000
        if parent_gui_node is None:
            gui_node = GuiNode(DEFAULT_WIDTH, DEFAULT_HEIGTH,
                               m.width // 2, m.height // 4,
                               m.width,
                               node, parent_gui_node, m,
                               self.rate, self.cv, self.elements, self.sketch, info)

        else:
            peers = [c for c in node.parent.get_children()]
            width_for_children = parent_gui_node.width_for_children // len(peers)
            i = peers.index(node)

            peer_width = parent_gui_node.width_for_children
            x = parent_gui_node.x - peer_width // 2 + peer_width // (len(peers) + 1) * (i + 1)
            y = parent_gui_node.y + DEFAULT_LAYERH_HEIGHT

            gui_node = GuiNode(DEFAULT_WIDTH, DEFAULT_HEIGTH,
                               x, y,
                               width_for_children,
                               node, parent_gui_node, m,
                               self.rate, self.cv, self.elements, self.sketch, info)
        return gui_node

    def generate_tree(self, m, info=False):
        if len(self.tree) > 0:
            tree_dict = {}
            tree_dict[None] = None
            q = deque()
            root_node = self.tree[0]
            q.append(root_node)
            while len(q) > 0:
                n = q.popleft()
                gui_node = self.generate_gui_node(n, tree_dict[n.parent], m, info)
                tree_dict[n] = gui_node

                children = n.get_children()
                for c in children:
                    if c is not None:
                        q.append(c)


class Map:
    def __init__(self, x=0, y=0, rate=0.1, cv=None, elements=None, sketch=None):
        self.width = 100000
        self.height = 100000
        self.cv = cv
        self.refer_x = x - self.width * rate // 2
        self.refer_y = y - self.height * rate // 2
        self.rect = cv.create_rectangle(x - self.width * rate // 2, y - self.height * rate // 2,
                                        x + self.width * rate // 2, y + self.height * rate // 2,
                                        outline='black',
                                        fill='white',
                                        )

        elements.append(self)
        sketch.append(self.rect)

    def delete_self(self):
        self.cv.delete(self.rect)


class GuiNode:
    def __init__(self,
                 width_on_map, height_on_map,
                 x_on_map, y_on_map,
                 width_for_children,
                 node, parent_gui_node, m,
                 rate, cv,
                 elements=None,
                 sketch=None, info=False):

        self.node = node
        self.cv = cv
        self.color = 'red' if node.mark else 'black'
        self.line = None
        self.info = None


        if parent_gui_node is not None:
            self.layer = parent_gui_node.layer + 1
            self.line = cv.create_line(m.refer_x + x_on_map * rate,
                                       m.refer_y + (y_on_map - height_on_map) * rate,
                                       m.refer_x + parent_gui_node.x * rate,
                                       m.refer_y + (parent_gui_node.y + height_on_map) * rate,
                                       fill=self.color)
            sketch.append(self.line)
        else:
            self.layer = 0
        width_on_map = width_on_map/(self.layer+1)
        height_on_map = height_on_map/(self.layer+1)
        self.rect = cv.create_rectangle(m.refer_x + (x_on_map - width_on_map + self.layer // 2) * rate,
                                        m.refer_y + (y_on_map - height_on_map + self.layer // 2) * rate,
                                        m.refer_x + (x_on_map + width_on_map - self.layer // 2) * rate,
                                        m.refer_y + (y_on_map + height_on_map - self.layer // 2) * rate,
                                        outline=self.color,
                                        fill='white' if self.layer%2==0 else 'black',
                                        )
        if info:
            self.info = cv.create_text(m.refer_x + (x_on_map - width_on_map + self.layer // 2) * rate,
                                       m.refer_y + (y_on_map - height_on_map - 40 + self.layer // 2) * rate,
                                       font=font.Font(size=int(10)),
                                       fill='red',
                                       text=node.info())
            sketch.append(self.info)

        self.x = x_on_map
        self.y = y_on_map
        self.width_for_children = width_for_children

        sketch.append(self.rect)

        elements.append(self)



    def delete_self(self):

        self.cv.delete(self.rect)
        if self.line is not None:
            self.cv.delete(self.line)
        if self.info is not None:
            self.cv.delete(self.info)


if __name__ == '__main__':
    # tree = generate_tree()
    # print(len(tree))
    # g = Gui(tree)
    pass
