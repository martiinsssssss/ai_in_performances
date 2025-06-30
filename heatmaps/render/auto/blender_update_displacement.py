import bpy

obj = bpy.context.active_object

if not obj or obj.type != 'MESH':
    print("Select an active mesh object.")
else:
    mat = obj.active_material
    if not mat or not mat.use_nodes:
        print("Object has no material with nodes.")
    else:
        print(f"Material: {mat.name}")
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        principled = None
        
        for node in nodes:
            print(f" - Node: {node.name} type: {node.type}")
            if node.type == 'BSDF_PRINCIPLED':
                principled = node

        if principled:
            base_color_links = principled.inputs['Base Color'].links
            print(f"Base Color links: {len(base_color_links)}")
            for link in base_color_links:
                print(f"  From {link.from_node.name} output {link.from_socket.name}")

            displacement_links = []
            for node in nodes:
                if node.type == 'OUTPUT_MATERIAL':
                    disp_links = node.inputs['Displacement'].links
                    displacement_links.extend(disp_links)
                    print(f"Displacement links in Material Output: {len(disp_links)}")
                    for link in disp_links:
                        print(f"  From {link.from_node.name} output {link.from_socket.name}")
        else:
            print("No Principled BSDF node found.")