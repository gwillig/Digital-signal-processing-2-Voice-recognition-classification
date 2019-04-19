from tkinter import ttk

def styleElements():
    style = ttk.Style()
    style.element_create("Custom.Treeheading.border", "from", "default")
    style.layout("Custom.Treeview.Heading", [
        ("Custom.Treeheading.cell", {'sticky': 'nswe'}),
        ("Custom.Treeheading.border", {'sticky': 'nswe', 'children': [
            ("Custom.Treeheading.padding", {'sticky': 'nswe', 'children': [
                ("Custom.Treeheading.image", {'side': 'right', 'sticky': ''}),
                ("Custom.Treeheading.text", {'sticky': 'we'})
            ]})
        ]}),
    ])

    style.configure("Custom.Treeview.Heading",
                    background="#18384e", foreground="#FFFFFF", relief="flat", font=('Helvetica', 10, "bold"))
    style.configure("Custom.Treeview",
                    background="#dddddd", foreground="#18384e", relief="flat")
    style.map("Custom.Treeview.Heading",
              relief=[('active', 'groove'), ('pressed', 'sunken')])
