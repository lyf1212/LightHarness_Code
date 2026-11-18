import bpy
import math
import random
from mathutils import Vector


def build_scene(ctx):
    rng = random.Random(1041)
    X, Y, Z = Vector((1,0,0)), Vector((0,1,0)), Vector((0,0,1))
    fx, fy = 922.1543194980696, 907.8845535714286
    cx, cy = 384.2413127413128, 256.2314285714286

    def P(u,v,depth):
        return Vector(((u-cx)*depth/fx,depth,(cy-v)*depth/fy))

    def material(name,color,rough=0.7,metal=0.0):
        return ctx.material(name,color,roughness=rough,metallic=metal)

    def glass(name,color,rough,ior):
        m = material(name,color,rough)
        nodes, links = m.node_tree.nodes, m.node_tree.links
        bs = nodes.get('Principled BSDF')
        bs.inputs['Transmission Weight'].default_value = 1.0
        bs.inputs['IOR'].default_value = ior
        path = nodes.new('ShaderNodeLightPath')
        transparent = nodes.new('ShaderNodeBsdfTransparent')
        transparent.inputs['Color'].default_value = (1,1,1,1)
        mix = nodes.new('ShaderNodeMixShader')
        links.new(path.outputs['Is Shadow Ray'],mix.inputs[0])
        links.new(bs.outputs['BSDF'],mix.inputs[1])
        links.new(transparent.outputs['BSDF'],mix.inputs[2])
        links.new(mix.outputs[0],nodes.get('Material Output').inputs['Surface'])
        return m

    def mesh(name,vertices,faces,mat,oid,smooth=False):
        data = bpy.data.meshes.new(name+'_mesh')
        data.from_pydata([tuple(v) for v in vertices],[],faces)
        data.update()
        ob = bpy.data.objects.new(name,data)
        bpy.context.collection.objects.link(ob)
        ctx.register(ob,oid)
        if mat is not None:
            data.materials.append(mat)
        for face in data.polygons:
            face.use_smooth = smooth
        return ob

    def box(name,center,dims,mat,oid,axes=None,bevel=0):
        c = Vector(center)
        a,b,d = (X,Y,Z) if axes is None else axes
        w,dep,h = dims
        verts = [c+a*(ix*w/2)+b*(iy*dep/2)+d*(iz*h/2) for iz in (-1,1) for iy in (-1,1) for ix in (-1,1)]
        faces = [(0,2,3,1),(4,5,7,6),(0,1,5,4),(2,6,7,3),(0,4,6,2),(1,3,7,5)]
        ob = mesh(name,verts,faces,mat,oid)
        if bevel:
            mod = ob.modifiers.new('Soft manufactured edges','BEVEL')
            mod.width, mod.segments = bevel, 3
        return ob

    def line(name,points,radius,mat,oid,cyclic=False):
        data = bpy.data.curves.new(name+'_curve','CURVE')
        data.dimensions = '3D'
        data.resolution_u = 2
        data.bevel_depth = radius
        data.bevel_resolution = 2
        data.use_fill_caps = True
        spline = data.splines.new('POLY')
        spline.points.add(len(points)-1)
        for p,co in zip(spline.points,points):
            p.co = (co[0],co[1],co[2],1)
        spline.use_cyclic_u = cyclic
        ob = bpy.data.objects.new(name,data)
        bpy.context.collection.objects.link(ob)
        data.materials.append(mat)
        ctx.register(ob,oid)
        return ob

    def rod(name,a,b,radius,mat,oid,radius2=None,sides=12):
        a,b = Vector(a),Vector(b)
        vec = b-a
        bpy.ops.mesh.primitive_cone_add(vertices=sides,radius1=radius,radius2=radius if radius2 is None else radius2,depth=vec.length,location=(a+b)/2)
        ob = bpy.context.object
        ob.name = name
        ob.rotation_mode = 'QUATERNION'
        ob.rotation_quaternion = Z.rotation_difference(vec.normalized())
        ob.data.materials.append(mat)
        for p in ob.data.polygons:
            p.use_smooth = len(p.vertices)==4
        ctx.register(ob,oid)
        return ob

    def ellipsoid(name,center,radii,mat,oid,direction=None,segments=16,rings=8):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=segments,ring_count=rings,radius=1,location=center)
        ob = bpy.context.object
        ob.name, ob.scale = name, radii
        if direction is not None:
            ob.rotation_mode = 'QUATERNION'
            ob.rotation_quaternion = X.rotation_difference(Vector(direction).normalized())
        ob.data.materials.append(mat)
        for p in ob.data.polygons:
            p.use_smooth = True
        ctx.register(ob,oid)
        return ob

    def rounded_points(w,h,r,count=8):
        pts = []
        for px,py,start in [(w/2-r,h/2-r,0),(-w/2+r,h/2-r,90),(-w/2+r,-h/2+r,180),(w/2-r,-h/2+r,270)]:
            for k in range(count+1):
                t = math.radians(start+90*k/count)
                pts.append((px+r*math.cos(t),py+r*math.sin(t)))
        return pts

    def plate(name,center,w,h,radius,thickness,horizontal,vertical,mat,oid):
        c = Vector(center)
        horizontal,vertical = Vector(horizontal),Vector(vertical)
        normal = horizontal.cross(vertical).normalized()
        outline = rounded_points(w,h,min(radius,w/2,h/2))
        verts = [c+horizontal*u+vertical*v+normal*offset for offset in (-thickness/2,thickness/2) for u,v in outline]
        n = len(outline)
        faces = [tuple(reversed(range(n))),tuple(range(n,2*n))]
        for i in range(n):
            j = (i+1)%n
            faces.append((i,j,j+n,i+n))
        return mesh(name,verts,faces,mat,oid)

    def difference(target,cutter):
        mod = target.modifiers.new('True recessed opening','BOOLEAN')
        mod.operation, mod.solver, mod.object = 'DIFFERENCE','EXACT',cutter
        bpy.context.view_layer.objects.active = target
        target.select_set(True)
        bpy.ops.object.modifier_apply(modifier=mod.name)
        target.select_set(False)
        cutter.hide_render = True
        cutter.hide_set(True)

    def lathe(name,center,profile,mat,oid,count=64,closed=False):
        c = Vector(center)
        verts = []
        for r,z in profile:
            for j in range(count):
                t = 2*math.pi*j/count
                verts.append(c+Vector((r*math.cos(t),r*math.sin(t),z)))
        faces = []
        for k in range(len(profile)-1):
            for j in range(count):
                q = (j+1)%count
                faces.append((k*count+j,k*count+q,(k+1)*count+q,(k+1)*count+j))
        if closed:
            faces.append(tuple(reversed(range(count))))
            faces.append(tuple((len(profile)-1)*count+j for j in range(count)))
        return mesh(name,verts,faces,mat,oid,True)

    wallmat = material('Charcoal plaster, moderate intrinsic reflectance',(0.0462,0.0511,0.0553),0.88)
    frame = material('Black anodized window metal',(0.012,0.017,0.019),0.42,0.25)
    reveal = material('Window reveal charcoal',(0.035,0.041,0.043),0.72)
    glazing = glass('Clear architectural glazing',(0.985,0.997,1.0),0.035,1.46)
    acrylic = glass('Clear blue-edged acrylic',(0.88,0.97,1.0),0.035,1.49)
    lidmat = glass('Thin clear PET lid',(0.98,0.99,1.0),0.06,1.47)
    strawmat = glass('Translucent straw plastic',(0.85,0.91,0.86),0.17,1.43)
    cabinetmat = material('Warm light gray molded cabinet',(0.49,0.47,0.415),0.52)
    recessmat = material('Deep charcoal recess liners',(0.009,0.014,0.015),0.55)
    slotmat = material('Dark slot surrounds',(0.026,0.04,0.043),0.35,0.35)
    slotinsert = material('Charging pack edge metal',(0.29,0.34,0.35),0.28,0.52)
    brown = material('Brown cup sleeve',(0.075,0.046,0.031),0.73)
    cupband = material('Cup lower dark band',(0.013,0.015,0.013),0.5)
    cupink = material('Faded cup print',(0.16,0.145,0.115),0.75)
    soil = material('Potting soil',(0.034,0.025,0.016),0.95)
    potmat = material('Dark olive green plant pot',(0.075,0.125,0.025),0.58)
    stemmat = material('Succulent woody green stems',(0.055,0.068,0.027),0.78)
    leaves = [material('Succulent leaf tone '+str(i),c,0.48) for i,c in enumerate([(0.045,0.10,0.025),(0.075,0.13,0.037),(0.033,0.073,0.02),(0.10,0.145,0.044)])]
    iron = material('Outdoor black steel',(0.022,0.03,0.031),0.36,0.45)
    railmat = material('Pale railing metal',(0.43,0.49,0.48),0.44,0.35)
    paving = material('Exterior pale paving',(0.44,0.43,0.38),0.91)
    jointmat = material('Paving joints',(0.20,0.22,0.21),0.96)
    ivory = material('Facade pale stone ribs',(0.66,0.65,0.56),0.72)
    ivorybright = material('Facade trim stone',(0.76,0.75,0.67),0.64)
    blueglass = [material('Facade blue glazing '+str(i),c,0.29,0.22) for i,c in enumerate([(0.18,0.34,0.49),(0.235,0.41,0.57),(0.285,0.45,0.59),(0.16,0.29,0.39),(0.27,0.41,0.49),(0.205,0.37,0.54)])]
    windowedge = material('Distant window subframes',(0.37,0.44,0.44),0.45,0.25)
    cavitymat = material('Facade interior recess',(0.045,0.067,0.08),0.88)
    lobbyglass = material('Street level blue glazing',(0.14,0.30,0.34),0.37,0.15)
    shopdark = material('Street storefront dark fascia',(0.06,0.079,0.078),0.72)
    redtrim = material('Storefront red trim',(0.49,0.035,0.035),0.64)
    yellow = material('Booth golden yellow trim',(0.87,0.51,0.04),0.63)
    cream = material('Booth warm cream panels',(0.79,0.75,0.60),0.76)
    orange = material('Decoration orange edge',(0.83,0.33,0.045),0.63)
    pink = material('Decoration pale pink',(0.78,0.40,0.39),0.72)
    whitepole = material('White painted boom',(0.84,0.86,0.81),0.48)

    a,b = Vector((0.69,5.22,0)),Vector((2.65,2.38,0))
    tangent = (b-a).normalized()
    normal = tangent.cross(Z).normalized()
    box('Right interior wall',(a+b)/2,((b-a).length,0.14,5.6),wallmat,'obj_002',(tangent,normal,Z))
    for zz in (-1.92,2.33):
        box('Main aperture horizontal boundary',(-1.71,5.26,zz),(3.57,0.18,0.075),frame,'obj_003')
    box('Main aperture far left boundary',(-3.49,5.26,0.20),(0.075,0.18,4.25),frame,'obj_003')
    box('Main clear window pane',(-1.715,5.275,0.20),(3.56,0.005,4.18),glazing,'obj_004')
    for zz in (-1.92,2.33):
        box('Narrow aperture horizontal boundary',(0.385,5.26,zz),(0.47,0.18,0.075),frame,'obj_005')
    box('Narrow clear window pane',(0.383,5.275,0.20),(0.463,0.005,4.18),glazing,'obj_006')
    box('Central deep window mullion',(0.107,5.145,0.20),(0.088,0.25,4.55),frame,'obj_007',bevel=0.004)
    box('Central mullion edge strip',(0.149,5.00,0.20),(0.012,0.025,4.55),reveal,'obj_007')
    box('Right glazing jamb',(0.639,5.20,0.20),(0.067,0.22,4.55),frame,'obj_008',bevel=0.003)
    box('Right recessed reveal',(0.684,5.22,0.20),(0.043,0.24,4.55),reveal,'obj_008')
    line('Thin jamb gasket',[(0.617,5.075,-1.7),(0.609,5.075,1.75)],0.006,frame,'obj_008')

    yaw = 0.222
    U = Vector((math.cos(yaw),math.sin(yaw),0))
    V = Vector((-math.sin(yaw),math.cos(yaw),0))
    C = Vector((0.345,4.235,0))
    def W(u,v,z):
        return C+U*u+V*v+Z*z

    outline = rounded_points(1.16,1.10,0.065,12)
    n = len(outline)
    verts = [W(u,v,-1.72) for u,v in outline]
    for ring in range(9):
        t = (math.pi/2)*ring/8
        for u,v in outline:
            weight = max(0.02,min(1.0,(-u-0.36)/0.22))
            inset = 0.14*weight*(1-math.cos(t))
            zz = -0.14-0.075*weight*(1-math.sin(t))
            verts.append(W(u+inset,v,zz))
    faces = [tuple(reversed(range(n)))]
    for ring in range(9):
        for i in range(n):
            j = (i+1)%n
            faces.append((ring*n+i,ring*n+j,(ring+1)*n+j,(ring+1)*n+i))
    faces.append(tuple(range(9*n,10*n)))
    cabinet = mesh('Rounded rental kiosk cabinet',verts,faces,cabinetmat,'obj_009')
    handle_u = 0.075
    cut = plate('Front recess construction cutter',W(handle_u,-0.55,-0.335),0.270,0.095,0.047,0.19,U,Z,None,'obj_009')
    difference(cabinet,cut)
    plate('Front handle recessed backing',W(handle_u,-0.469,-0.335),0.266,0.091,0.045,0.007,U,Z,recessmat,'obj_009')
    for row,z in enumerate((-0.505,-0.69,-0.875,-1.06)):
        for col,v in enumerate((-0.245,0.31)):
            suffix = str(row)+' '+str(col)
            cut = plate('Slot cutter '+suffix,W(-0.58,v,z),0.305,0.101,0.035,0.15,V,Z,None,'obj_009')
            difference(cabinet,cut)
            plate('Slot inner surround '+suffix,W(-0.518,v,z),0.298,0.096,0.03,0.015,V,Z,slotmat,'obj_009')
            plate('Slot deep dark well '+suffix,W(-0.529,v,z),0.255,0.062,0.017,0.003,V,Z,recessmat,'obj_009')
            box('Power bank silver lip '+suffix,W(-0.535,v,z-0.008),(0.016,0.228,0.024),slotinsert,'obj_009',(U,V,Z),0.004)
            box('Power bank seam '+suffix,W(-0.545,v,z+0.02),(0.009,0.205,0.006),slotmat,'obj_009',(U,V,Z))
    original_vertices = [v.co.copy() for v in cabinet.data.vertices]
    top_faces, side_faces = [],[]
    for p in cabinet.data.polygons:
        is_top = p.normal.z>0.999 and all(abs(original_vertices[i].z+0.14)<0.00001 for i in p.vertices)
        if is_top:
            top_faces.append(tuple(p.vertices))
        else:
            side_faces.append(tuple(p.vertices))
    side_data = bpy.data.meshes.new('Kiosk shell without duplicate top')
    side_data.from_pydata(original_vertices,[],side_faces)
    side_data.materials.append(cabinetmat)
    side_data.update()
    cabinet.data = side_data
    mesh('Kiosk upper support surface',original_vertices,top_faces,cabinetmat,'obj_010')
    badge = material('Muted cyan kiosk badge',(0.12,0.40,0.43),0.36,0.15)
    ellipsoid('Small side badge',W(-0.585,-0.225,-0.279),(0.005,0.02,0.06),badge,'obj_009',segments=24,rings=12)
    line('Front molded seam',[W(-0.46,-0.552,-0.55),W(0.50,-0.552,-0.55)],0.0013,slotmat,'obj_009')

    labelbg = material('Deep green label border',(0.012,0.032,0.018),0.78)
    labelink = material('Muted green label lettering',(0.19,0.26,0.16),0.78)
    goldink = material('Small gold payment emblem',(0.30,0.22,0.075),0.64)
    glyphs = [
      [[(0.30,1),(0.05,0.7)],[(0.30,0.73),(0.04,0.41)],[(0.18,0.55),(0.18,0)],[(0.44,0.88),(0.44,0.7),(0.84,0.7),(0.84,0.88)],[(0.64,1),(0.64,0.67)],[(0.39,0.56),(0.87,0.56)],[(0.48,0.55),(0.42,0.07)],[(0.65,0.54),(0.64,0.05)],[(0.79,0.52),(0.83,0.2)],[(0.98,0.87),(0.86,0.54),(0.94,0.12)],[(0.89,0.6),(0.74,0.02)]],
      [[(0.29,0.97),(0.06,0.61)],[(0.18,0.76),(0.18,0)],[(0.60,1),(0.70,0.91)],[(0.38,0.84),(0.99,0.84)],[(0.46,0.68),(0.92,0.68)],[(0.46,0.52),(0.92,0.52)],[(0.46,0.08),(0.46,0.34),(0.92,0.34),(0.92,0.08),(0.46,0.08)]],
      [[(0.39,1),(0.15,0.75)],[(0.33,0.89),(0.77,0.89),(0.62,0.7)],[(0.23,0.69),(0.23,0.42),(0.88,0.42),(0.88,0.69),(0.23,0.69)],[(0.55,0.68),(0.55,0.43)],[(0.45,0.42),(0.30,0.1),(0.04,0.02)],[(0.67,0.42),(0.67,0.06),(0.97,0.06),(0.97,0.24)]],
      [[(0.05,0.72),(0.39,0.72)],[(0.25,1),(0.25,0.04),(0.10,0.02)],[(0.02,0.4),(0.38,0.58)],[(0.48,0.83),(0.94,0.83),(0.94,0.32),(0.48,0.32),(0.48,0.83)],[(0.49,0.58),(0.93,0.58)],[(0.72,0.85),(0.72,0)]],
      [[(0.04,0.67),(0.49,1),(0.98,0.68)],[(0.26,0.68),(0.77,0.68)],[(0.15,0.48),(0.91,0.48)],[(0.52,0.67),(0.52,0.08)],[(0.25,0.37),(0.34,0.18)],[(0.80,0.38),(0.68,0.19)],[(0.07,0.06),(0.96,0.06)]],
      [[(0.29,1),(0.04,0.62)],[(0.18,0.77),(0.18,0)],[(0.36,0.8),(0.97,0.8)],[(0.65,1),(0.65,0.25),(0.46,0.04)],[(0.40,0.61),(0.92,0.61),(0.92,0.37),(0.40,0.37),(0.40,0.61)],[(0.39,0.29),(0.64,0.08),(0.97,0.02)]],
      [[(0.17,0.02),(0.23,0.3),(0.23,0.92),(0.93,0.92),(0.93,0.03),(0.78,0.03)],[(0.24,0.64),(0.92,0.64)],[(0.24,0.36),(0.92,0.36)],[(0.59,0.91),(0.59,0.08)]],
      [[(0.49,1),(0.60,0.87)],[(0.08,0.80),(0.97,0.80)],[(0.49,0.79),(0.29,0.5),(0.85,0.52)],[(0.72,0.67),(0.9,0.44)],[(0.41,0.43),(0.31,0.12),(0.04,0.03)],[(0.67,0.44),(0.67,0.05),(0.96,0.05),(0.98,0.23)]],
      [[(0.15,0.84),(0.91,0.84),(0.91,0.34),(0.15,0.34),(0.15,0.84)],[(0.15,0.60),(0.9,0.60)],[(0.54,1),(0.54,0.06),(0.95,0.06),(0.99,0.22)]],
      [[(0.50,1),(0.60,0.89)],[(0.10,0.72),(0.10,0.86),(0.94,0.86),(0.94,0.72)],[(0.25,0.66),(0.83,0.66)],[(0.27,0.43),(0.82,0.43)],[(0.55,0.66),(0.55,0.1)],[(0.75,0.32),(0.84,0.19)],[(0.12,0.07),(0.97,0.07)]]
    ]
    for row in range(2):
        top = -0.704-row*0.175
        for col in range(5):
            left = -0.22+col*0.121
            for j,stroke in enumerate(glyphs[row*5+col]):
                suffix = str(row)+' '+str(col)+' '+str(j)
                borderpoints = [W(left+u*0.108,-0.555,top-(1-v)*0.12) for u,v in stroke]
                inkpoints = [W(left+u*0.108,-0.565,top-(1-v)*0.12) for u,v in stroke]
                line('Contour-following glyph backing '+suffix,borderpoints,0.0092,labelbg,'obj_009')
                line('Label stroke '+suffix,inkpoints,0.0034,labelink,'obj_009')
    ellipsoid('Payment mark',W(-0.07,-0.559,-0.598),(0.025,0.004,0.025),goldink,'obj_009')
    for k in range(17):
        u = -0.023+k*0.018
        line('Tiny payment word mark '+str(k),[W(u,-0.559,-0.585),W(u,-0.559,-0.613)],0.0017,labelink,'obj_009')
        if k%2==0:
            line('Tiny payment cross stroke '+str(k),[W(u-0.005,-0.560,-0.598),W(u+0.007,-0.560,-0.598)],0.0014,labelink,'obj_009')

    cupcenter = Vector((-0.070,4.65,0))
    lathe('Tapered brown takeaway cup',cupcenter,[(0.132,-0.14),(0.138,-0.127),(0.179,0.205),(0.180,0.221),(0.173,0.222),(0.170,0.196)],brown,'obj_011',64,True)
    lathe('Cup black lower band',cupcenter,[(0.133,-0.141),(0.141,-0.095)],cupband,'obj_011',64)
    rod('Drink dark top surface',cupcenter+Z*0.208,cupcenter+Z*0.211,0.171,brown,'obj_011',sides=64)
    lathe('Cup upper rolled lip',cupcenter,[(0.177,0.211),(0.185,0.213),(0.188,0.220),(0.184,0.228),(0.177,0.226)],lidmat,'obj_012',80)
    profile = []
    for k in range(19):
        angle = (math.pi/2-0.08)*k/18
        profile.append((0.187*math.cos(angle),0.224+0.161*math.sin(angle)))
    for k in reversed(range(19)):
        angle = (math.pi/2-0.08)*k/18
        profile.append((0.184*math.cos(angle),0.224+0.158*math.sin(angle)))
    lathe('Hollow transparent dome lid',cupcenter,profile,lidmat,'obj_012',80)
    lathe('Lid lower retaining ring',cupcenter,[(0.184,0.210),(0.194,0.213),(0.195,0.219),(0.190,0.224)],lidmat,'obj_012',80)
    strawbase,strawtop = Vector((-0.051,4.647,0.362)),Vector((-0.032,4.655,0.578))
    axis = (strawtop-strawbase).normalized()
    e1 = axis.cross(Y).normalized()
    e2 = axis.cross(e1).normalized()
    sv = []
    for end,rad in [(strawbase,0.011),(strawtop,0.011),(strawtop,0.0075),(strawbase,0.0075)]:
        for j in range(24):
            t = j*2*math.pi/24
            sv.append(end+rad*(e1*math.cos(t)+e2*math.sin(t)))
    sf = []
    for k in range(4):
        nk = (k+1)%4
        for j in range(24):
            q = (j+1)%24
            sf.append((k*24+j,k*24+q,nk*24+q,nk*24+j))
    mesh('Open ended drink straw',sv,sf,strawmat,'obj_013',True)
    line('Subtle cup printed initial',[(-0.003,4.476,0.148),(-0.007,4.480,0.057),(0.025,4.483,0.060),(0.039,4.487,0.085),(0.009,4.478,0.101),(0.037,4.486,0.127),(0.023,4.482,0.149),(-0.003,4.476,0.148)],0.004,cupink,'obj_011')

    an = Vector((0.6,0.8,0.0))
    plane_d = an.dot(Vector((0.076,4.98,0)))
    av = []
    for u,v in [(398.3,282.3),(459.1,284.0),(457.0,114.0),(398.3,165.1)]:
        ray = P(u,v,1.0)
        av.append(ray*(plane_d/an.dot(ray)))
    pv = [v-an*0.004 for v in av]+[v+an*0.004 for v in av]
    pf = [(0,1,2,3),(7,6,5,4),(1,0,4,5),(2,1,5,6),(3,2,6,7),(0,3,7,4)]
    panel = mesh('Clear acrylic upright panel',pv,pf,acrylic,'obj_014')
    line('Acrylic upper polished edge',[av[3],av[2]],0.003,acrylic,'obj_014')
    line('Acrylic right polished edge',[av[1],av[2]],0.0035,acrylic,'obj_014')
    footcenter = (av[0]+av[1])/2
    footcenter.z = -0.138
    panelwidth = Vector((0.8,-0.6,0))
    box('Acrylic foot',footcenter,(0.40,0.18,0.009),acrylic,'obj_014',(panelwidth,an,Z),bevel=0.004)

    pc = Vector((0.692,4.30,0))
    potprofile = [(0.179,-0.14),(0.188,-0.129),(0.269*0.92,0.257),(0.278*0.92,0.272),(0.276*0.92,0.283),(0.258*0.92,0.286),(0.252*0.92,0.262),(0.228*0.92,0.162)]
    lathe('Tapered green plant pot',pc,potprofile,potmat,'obj_015',80,True)
    lathe('Plant pot rim',pc,[(0.265*0.92,0.266),(0.282*0.92,0.270),(0.283*0.92,0.281),(0.276*0.92,0.290),(0.262*0.92,0.288)],potmat,'obj_015',80)
    rod('Recessed soil surface',pc+Z*0.244,pc+Z*0.250,0.245*0.92,soil,'obj_015',sides=48)
    roots = pc+Vector((0,0,0.235))
    frng = random.Random(2729)
    endpoints = [(0.445,4.29,0.525),(0.505,4.38,0.565),(0.972,4.40,0.542),(0.622,4.38,0.524),(0.698,4.42,0.489),(0.772,4.37,0.474),(0.863,4.34,0.647),(0.930,4.42,0.552),(0.970,4.32,0.455),(0.988,4.27,0.387),(0.455,4.25,0.362),(0.540,4.18,0.413),(0.742,4.29,0.546),(0.956,4.31,0.502)]
    for bi,endco in enumerate(endpoints):
        end = Vector(endco)
        start = roots+Vector((frng.uniform(-0.055,0.055),frng.uniform(-0.045,0.045),0))
        foot = Vector((end.x+frng.uniform(-0.028,0.028),end.y+frng.uniform(-0.02,0.02),0.284+frng.uniform(-0.008,0.028)))
        line('Succulent upright branch '+str(bi),[start,start.lerp(foot,0.52)+Z*0.025,foot,end],0.0052 if bi!=6 else 0.0075,stemmat,'obj_016')
        for node in range(3,15):
            t = node/15
            pos = foot.lerp(end,t)
            pos.x += 0.007*math.sin(t*math.pi*2+bi)
            for side in (-1,1):
                angle = bi*0.71+node*1.24+(0 if side==1 else math.pi)
                direction = Vector((math.cos(angle),0.72*math.sin(angle),frng.uniform(0.05,0.48))).normalized()
                length = frng.uniform(0.027,0.041)*(0.82 if node>12 else 1.0)
                width = frng.uniform(0.013,0.019)
                thick = frng.uniform(0.005,0.008)
                leafmat = leaves[frng.randrange(4)]
                omit = bi in (0,1,10,11) and node in (4,8,11) and side == (1 if bi%2 else -1)
                if omit:
                    continue
                leafcenter = pos+direction*length*0.77
                suffix = str(bi)+' '+str(node)+' '+str(side)
                rod('Leaf petiole '+suffix,pos,leafcenter,0.0016,stemmat,'obj_016',sides=6)
                ellipsoid('Succulent oval leaf '+suffix,leafcenter,(length,width,thick),leafmat,'obj_016',direction,12,6)
        for k in range(3):
            direction = Vector((frng.uniform(-1,1),frng.uniform(-0.7,0.7),0.4)).normalized()
            ellipsoid('Branch tip leaf '+str(bi)+' '+str(k),end+direction*0.013,(0.020,0.011,0.006),leaves[(bi+k)%4],'obj_016',direction,12,6)
    for shoot,endco in enumerate([(0.354,4.22,0.155),(0.401,4.17,0.105)]):
        elbow = Vector((0.425+shoot*0.018,4.21,0.338))
        end = Vector(endco)
        line('Descending left succulent shoot '+str(shoot),[roots,Vector((0.49,4.24,0.325)),elbow,elbow.lerp(end,0.45)-X*0.018,end],0.0047,stemmat,'obj_016')
        for node in range(11):
            t = (node+1+0.18*math.sin(node*2.1+shoot))/12
            pos = elbow.lerp(end,t)
            for side in (-1,1):
                direction = Vector((side*0.8,frng.uniform(-0.5,0.5),frng.uniform(-0.35,0.3))).normalized()
                length = frng.uniform(0.022,0.032)
                if (node+3*shoot)%4==0 and side==1:
                    continue
                direction = (direction+Vector((0.12*math.sin(node*1.7),0.18*math.cos(node*2.3),0.22*math.sin(node+shoot)))).normalized()
                leafcenter = pos+direction*length*0.65
                suffix = str(shoot)+' '+str(node)+' '+str(side)
                rod('Trailing leaf petiole '+suffix,pos,leafcenter,0.0015,stemmat,'obj_016',sides=6)
                ellipsoid('Trailing oval leaf '+suffix,leafcenter,(length,0.013,0.0055),leaves[(node+shoot)%4],'obj_016',direction,12,6)

    def groundz(y):
        return -2.0-0.087*y
    mesh('Exterior paving plane',[(-18,7,groundz(7)),(12,7,groundz(7)),(12,35,groundz(35)),(-18,35,groundz(35))],[(0,1,2,3)],paving,'obj_001')
    for j in range(8,25,2):
        line('Paving transverse joint '+str(j),[(-12,j,groundz(j)+0.001),(8,j,groundz(j)+0.001)],0.007,jointmat,'obj_001')
    for j in range(-12,8,2):
        line('Paving longitudinal joint '+str(j),[(j,7,groundz(7)+0.001),(j,26,groundz(26)+0.001)],0.005,jointmat,'obj_001')
    tc = P(32,480,8.05)
    table_normal = Vector((0,0.164,1)).normalized()
    rod('Thin round cafe tabletop',tc-table_normal*0.024,tc,0.69,iron,'obj_018',sides=64)
    rod('Table pedestal',tc-Z*0.64,tc-table_normal*0.03,0.032,iron,'obj_017')
    for angle in (0.4,2.5,4.6):
        foot = tc+Vector((0.52*math.cos(angle),0.52*math.sin(angle),-0.76))
        rod('Splayed table foot',tc-Z*0.52,foot,0.022,iron,'obj_017')

    def chair(oid,name,u,depth,width,yaw_ch):
        center = P(u,477,depth)
        ax = Vector((math.cos(yaw_ch),math.sin(yaw_ch),0))
        ay = Vector((-math.sin(yaw_ch),math.cos(yaw_ch),0))
        revised = oid == 'obj_020'
        top = 0.37
        def Q(a,b,z):
            lean = -min(0.20,max(0,top-z)*0.26) if revised else 0.0
            return center+ax*a+ay*b+Z*z+X*lean
        for s in (-1,1):
            pts = [Q(s*width*0.55,-0.03,-0.70),Q(s*width*0.49,0.05,-0.16),Q(s*width*0.47,0.11,top-0.07),Q(s*width*0.40,0.13,top)]
            line(name+' curved upright '+str(s),pts,0.019,iron,oid)
            rod(name+' forward leg '+str(s),Q(s*width*0.44,-0.52,-0.72),Q(s*width*0.43,-0.39,-0.19),0.019,iron,oid)
        for r,h in enumerate((top-0.015,top-0.12,top-0.235,-0.09)):
            points = []
            for k in range(17):
                t = -1+2*k/16
                bow = 0.22 if revised else 0.11
                sag = 0.04 if revised else 0.027
                points.append(Q(t*width*0.44,0.10+bow*(1-t*t),h-sag*(1-t*t)))
            line(name+' bowed back rail '+str(r),points,0.015,iron,oid)
        corners = [Q(-width*0.45,-0.46,-0.20),Q(width*0.45,-0.46,-0.20),Q(width*0.46,0.12,-0.20),Q(-width*0.46,0.12,-0.20)]
        line(name+' seat rim',corners,0.023,iron,oid,True)
        for k in range(7):
            yy = -0.42+k*0.078
            rod(name+' seat slat '+str(k),Q(-width*0.41,yy,-0.195),Q(width*0.41,yy,-0.195),0.015,iron,oid)
        rod(name+' diagonal brace one',Q(-width*0.44,0.12,-0.48),Q(width*0.44,0.12,-0.12),0.012,iron,oid)
        rod(name+' diagonal brace two',Q(width*0.44,0.12,-0.48),Q(-width*0.44,0.12,-0.12),0.012,iron,oid)
    chair('obj_019','Left cafe chair',112,8.65,0.97,-0.12)
    chair('obj_020','Right cafe chair',166,9.30,0.98,0.16)
    raildepth = 10.5
    for row in (430,468):
        rod('Exterior railing horizontal '+str(row),P(207,row,raildepth),P(337,row,raildepth),0.020,railmat,'obj_021')
    for k,u in enumerate(range(210,337,10)):
        rod('Exterior railing upright '+str(k),P(u,430,raildepth),P(u,527,raildepth),0.015,railmat,'obj_021')
    for u in (210,275,336):
        rod('Exterior railing structural post',P(u,428,raildepth),P(u,535,raildepth),0.027,railmat,'obj_021')
    line('Physical white diagonal boom',[P(122,420,12.0),P(152,385,12.05),P(187,353,12.1),P(220,329,12.15),P(248,313,12.2)],0.042,whitepole,'obj_022')

    bc = P(280,432,13.8)
    bw,bdepth,bheight = 1.23,0.87,1.62
    btop = P(280,374,13.8).z
    bottom = btop-bheight
    for sx in (-1,1):
        for sy in (-1,1):
            box('Booth corner post',(bc.x+sx*bw/2,13.8+sy*bdepth/2,(bottom+btop)/2),(0.077,0.077,bheight),cream,'obj_023',bevel=0.01)
    box('Booth lower front panel',(bc.x,13.8-bdepth/2,bottom+0.37),(bw,0.065,0.74),cream,'obj_023')
    box('Booth lower side panel',(bc.x+bw/2,13.8,bottom+0.37),(0.065,bdepth,0.74),cream,'obj_023')
    box('Booth recessed front glazing',(bc.x,13.8-bdepth/2+0.045,btop-0.43),(bw-0.14,0.008,0.75),glazing,'obj_023')
    box('Booth recessed back wall',(bc.x,13.8+bdepth/2,btop-0.63),(bw,0.075,1.1),cream,'obj_023')
    box('Booth side glazed opening',(bc.x+bw/2-0.035,13.8,btop-0.43),(0.008,bdepth-0.15,0.75),glazing,'obj_023')
    box('Booth broad yellow roof',(bc.x,13.8,btop+0.075),(1.65,1.19,0.15),yellow,'obj_023',bevel=0.065)
    box('Booth pale roof inset',(bc.x,13.75,btop+0.119),(1.43,1.0,0.075),cream,'obj_023',bevel=0.035)
    for xx in (-0.48,0.48):
        box('Booth golden side trim',(bc.x+xx,13.345,bottom+0.74),(0.038,0.025,1.25),yellow,'obj_023')
    box('Booth counter',(bc.x,13.30,btop-0.78),(1.37,0.22,0.075),cream,'obj_023',bevel=0.018)
    for u in (260,316):
        rod('Rooftop sign support',P(u,370,13.95),P(u,326,13.95),0.025,yellow,'obj_024')

    def sculpted_lobe(name,center,rx,depth,rz,tilt,phase,mat):
        c = Vector(center)
        sides, rings = 24, 18
        ct,st = math.cos(tilt),math.sin(tilt)
        vv = []
        for k in range(rings+1):
            t = k/rings
            profile = max(0.018,math.sin(math.pi*t)**0.62*(1.25-0.55*t))
            profile *= 1.0+0.065*math.sin(3*math.pi*t+phase)
            sweep = rx*0.22*math.sin(math.pi*t)*math.sin(phase+1.8*t)
            for j in range(sides):
                a = 2*math.pi*j/sides
                xx = rx*profile*math.cos(a)*(1+0.06*math.sin(a+phase))+sweep
                yy = depth*profile*math.sin(a)
                zz = rz*(2*t-1)
                vv.append(c+Vector((ct*xx+st*zz,yy,-st*xx+ct*zz)))
        ff = [tuple(reversed(range(sides)))]
        for k in range(rings):
            for j in range(sides):
                q = (j+1)%sides
                ff.append((k*sides+j,k*sides+q,(k+1)*sides+q,(k+1)*sides+j))
        ff.append(tuple(rings*sides+j for j in range(sides)))
        return mesh(name,vv,ff,mat,'obj_024',True)

    lobe_specs = [(264,327,0.145,0.445,-0.52,0.4),(285,321,0.193,0.465,-0.29,1.8),(308,317,0.151,0.412,-0.58,3.1),(329,310,0.182,0.342,-0.23,4.4)]
    for k,(u,v,rx,rz,ang,phase) in enumerate(lobe_specs):
        cp = P(u,v,14.0)
        sculpted_lobe('Golden sculpted sign petal '+str(k),cp,rx,0.12,rz,ang,phase,orange)
        inset_center = cp+Vector((-0.012 if k%2==0 else 0.012,-0.118,0.025))
        sculpted_lobe('Small inset sign petal '+str(k),inset_center,rx*(0.43 if k%2==0 else 0.51),0.022,rz*(0.59 if k%2==0 else 0.65),ang+0.04,phase,cream if k%2==0 else pink)
    for k,(u,v,r) in enumerate([(251,308,0.16),(267,299,0.14),(288,286,0.17),(307,294,0.19),(323,287,0.23),(340,294,0.19),(344,314,0.18)]):
        ellipsoid('Irregular yellow rooftop ornament '+str(k),P(u,v,14.12),(r,0.11,r*0.68),yellow,'obj_024',segments=20,rings=10)
    signgray = material('Back of exterior gray sign',(0.34,0.37,0.37),0.83)
    box('Exterior rectangular signboard',P(320,241,10.0),(0.56,0.075,0.87),signgray,'obj_025',bevel=0.012)
    rod('Signboard mounting pole',P(318,280,10.08),P(318,365,10.08),0.025,railmat,'obj_025')

    facade_shear = 0.085
    def F(x,y,z):
        return Vector((x+0.015*z,y,z+facade_shear*x))
    def facade_box(name,x,y,z,w,d,h,mat):
        vv = [F(x+ix*w/2,y+iy*d/2,z+iz*h/2) for iz in (-1,1) for iy in (-1,1) for ix in (-1,1)]
        ff = [(0,2,3,1),(4,5,7,6),(0,1,5,4),(2,6,7,3),(0,4,6,2),(1,3,7,5)]
        return mesh(name,vv,ff,mat,'obj_026')
    left,cols,span = -8.90,18,9.24
    step = span/cols
    base,floorheight,floors = 0.23,0.79,11
    for j in range(cols+1):
        xx = left+j*step
        facade_box('High rise vertical stone rib '+str(j),xx,19.98,base+floors*floorheight/2,0.073,0.25,floors*floorheight+0.10,ivory)
    for row in range(floors+1):
        zz = base+row*floorheight
        facade_box('High rise horizontal floor band '+str(row),left+cols*step/2,20.005,zz,cols*step+0.08,0.19,0.086,ivorybright)
    for col in range(cols):
        xx = left+(col+0.5)*step
        for row in range(floors):
            zz = base+(row+0.5)*floorheight
            suffix = str(col)+' '+str(row)
            facade_box('Recessed facade window '+suffix,xx,20.135,zz,step-0.078,0.018,floorheight-0.088,blueglass[rng.randrange(len(blueglass))])
            facade_box('Window internal vertical frame '+suffix,xx+0.027,20.107,zz,0.022,0.042,floorheight-0.1,windowedge)
            facade_box('Window internal transom '+suffix,xx,20.105,zz+0.03,step-0.085,0.04,0.022,windowedge)
            if (row+col)%4==0:
                facade_box('Recessed operable window inset '+suffix,xx-0.073,20.092,zz-0.17,0.09,0.012,0.16,blueglass[3])
    facade_box('Main facade backing shell',left+cols*step/2,20.50,base+floors*floorheight/2,cols*step+0.12,0.12,floors*floorheight,cavitymat)
    for k in range(18):
        xx = -6.50+k*0.161
        facade_box('Dense projecting vertical architectural fin '+str(k),xx,19.81,base+4.25,0.079,0.30,8.58,ivory)
    facade_box('Main facade heavy lower cornice',left+cols*step/2,19.86,base-0.015,cols*step+0.18,0.34,0.15,ivorybright)
    H,N = Vector((0.86,0.51,0)),Vector((-0.51,0.86,0))
    corner = Vector((0.35,20.0,0))
    for j in range(8):
        box('Return facade upright '+str(j),corner+H*(j*0.46)+Z*4.6,(0.082,0.20,9.0),ivorybright,'obj_026',(H,N,Z))
    for row in range(12):
        zz = 0.23+row*floorheight
        box('Return facade floor band '+str(row),corner+H*1.60+Z*zz,(3.29,0.24,0.095),ivorybright,'obj_026',(H,N,Z))
        if row<11:
            for j in range(7):
                box('Return recessed blue pane '+str(j)+' '+str(row),corner+H*((j+0.5)*0.46)+N*0.085+Z*(zz+floorheight/2),(0.378,0.014,floorheight-0.11),blueglass[(j+row)%6],'obj_026',(H,N,Z))
    box('Return facade interior shell',corner+H*1.6+N*0.32+Z*4.4,(3.3,0.10,9.0),cavitymat,'obj_026',(H,N,Z))
    for j in range(13):
        xx = -8.75+j*0.74
        top,bottom = 0.23+facade_shear*xx,-2.36
        box('Street level stone post '+str(j),(xx,19.91,(top+bottom)/2),(0.10,0.26,top-bottom),ivorybright,'obj_026')
        if j<12:
            midx = xx+0.37
            midtop = 0.23+facade_shear*midx
            box('Street level recessed blue pane '+str(j),(midx,20.095,(midtop+bottom)/2),(0.635,0.018,midtop-bottom-0.045),lobbyglass,'obj_026')
            box('Street level glass transom '+str(j),(midx,20.065,-1.13),(0.64,0.032,0.035),windowedge,'obj_026')
    box('Street level recessed shell',(-4.1,20.46,-1.15),(9.6,0.10,2.55),cavitymat,'obj_026')
    box('Dark storefront sign fascia',(-4.15,19.87,-2.595),(9.65,0.22,0.43),shopdark,'obj_026')
    box('Thin red storefront strip',(-4.15,19.725,-2.85),(9.66,0.055,0.068),redtrim,'obj_026')
    for j in range(15):
        xx = -8.85+j*0.64
        box('Ground shop doorway surround '+str(j),(xx,19.93,-3.35),(0.071,0.20,0.94),ivory,'obj_026')
        box('Ground shop glass recess '+str(j),(xx+0.32,20.12,-3.36),(0.565,0.014,0.96),blueglass[(j+3)%6],'obj_026')
    for k,u in enumerate((32,77,118,157,203,246,284)):
        cp = P(u,411,19.65)
        box('Small storefront display '+str(k),cp,(0.30,0.05,0.38),cream if k%3 else yellow,'obj_026')
        if k%2:
            box('Storefront display red detail '+str(k),cp+Vector((0,-0.033,0.08)),(0.26,0.015,0.065),redtrim,'obj_026')
    box('Golden storefront sign',P(125,374,19.58),(0.91,0.08,0.30),yellow,'obj_026',bevel=0.025)

    skin = material('Pedestrian skin',(0.43,0.29,0.21),0.82)
    hair = material('Pedestrian dark hair',(0.022,0.023,0.02),0.82)
    shirtwhite = material('Pedestrian light shirt',(0.70,0.73,0.70),0.86)
    lavender = material('Pedestrian pale lavender shirt',(0.49,0.43,0.55),0.85)
    bluecloth = material('Pedestrian blue gray shirt',(0.18,0.29,0.40),0.86)
    darkcloth = material('Pedestrian charcoal clothing',(0.034,0.046,0.058),0.9)
    tancloth = material('Pedestrian light trousers',(0.47,0.46,0.39),0.9)
    def person(oid,name,u,headv,footv,depth,shirt,trousers,pose=0):
        top,foot = P(u,headv,depth),P(u,footv,depth)
        height = top.z-foot.z
        r = height*0.073
        hc = top-Z*r
        ellipsoid(name+' head',hc,(r*0.76,r*0.74,r),skin,oid,segments=16,rings=10)
        ellipsoid(name+' hair',hc+Vector((0,0.016,r*0.34)),(r*0.81,r*0.77,r*0.73),hair,oid,segments=16,rings=8)
        shoulder = top-Z*(height*0.235)
        hip = foot+Z*(height*0.43)
        waist = (shoulder+hip)/2
        ellipsoid(name+' clothed torso',waist,(height*0.13,height*0.083,(shoulder.z-hip.z)*0.65),shirt,oid,segments=16,rings=10)
        for side in (-1,1):
            sh = shoulder+X*(side*height*0.13)
            elbow = sh+Vector((side*height*(0.035+0.025*pose),-0.025,-height*0.20))
            hand = elbow+Vector((-side*height*0.025,-0.015,-height*0.14))
            rod(name+' upper arm '+str(side),sh,elbow,height*0.041,shirt,oid,radius2=height*0.033)
            rod(name+' forearm '+str(side),elbow,hand,height*0.029,skin,oid,radius2=height*0.023)
            ellipsoid(name+' hand '+str(side),hand,(height*0.025,height*0.023,height*0.039),skin,oid,segments=12,rings=6)
            legstart = hip+X*(side*height*0.066)
            knee = foot+Vector((side*height*(0.07+0.025*pose),side*0.025,height*0.23))
            ankle = foot+Vector((side*height*(0.065+0.055*pose),-side*height*0.065,height*0.04))
            rod(name+' upper trouser leg '+str(side),legstart,knee,height*0.059,trousers,oid,radius2=height*0.043)
            rod(name+' lower trouser leg '+str(side),knee,ankle,height*0.043,trousers,oid,radius2=height*0.030)
            ellipsoid(name+' shoe '+str(side),ankle+Vector((0,-height*0.038,-height*0.02)),(height*0.047,height*0.079,height*0.031),darkcloth,oid,segments=12,rings=6)
    person('obj_027','Far left pedestrian',15,401,473,13.05,shirtwhite,darkcloth,1)
    person('obj_028','Left light clothed pedestrian',125,402,470,12.75,shirtwhite,tancloth,1)
    person('obj_029','Adjacent light clothed pedestrian',151,405,471,12.95,shirtwhite,tancloth,0)
    person('obj_030','Blue clothed pedestrian',180,404,466,13.0,bluecloth,darkcloth,1)
    person('obj_031','Central dark clothed pedestrian',214,391,472,12.65,darkcloth,darkcloth,0)
    person('obj_032','Lavender pedestrian at booth',257,406,476,13.05,lavender,darkcloth,0)
    person('obj_033','Pale pedestrian at booth',281,399,478,13.12,shirtwhite,darkcloth,0)

    # Diagnostics only: no geometry, material, camera or rendering changes.
    # Cast against the finite panel and scene, not merely an infinite mirror plane.
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene
    origin = Vector((0,0,0))
    front_d = plane_d-0.004
    panel['reflection_diagnostic_note'] = 'Geometric first-surface specular ray test; not a radiometric or transmission-pass measurement.'
    for k,height in enumerate((-0.06,0.06,0.16)):
        target = cupcenter+Z*height
        virtual = target+an*(2*(front_d-an.dot(target)))
        incoming = virtual.normalized()
        point = incoming*(front_d/an.dot(incoming))
        reflected = (incoming-2*incoming.dot(an)*an).normalized()
        first = scene.ray_cast(depsgraph,origin,incoming,distance=point.length+0.02)
        second = scene.ray_cast(depsgraph,point-an*0.001,reflected,distance=2.0)
        key = 'reflection_sample_'+str(k)
        panel[key+'_pixel'] = [float(cx+fx*virtual.x/virtual.y),float(cy-fy*virtual.z/virtual.y)]
        panel[key+'_camera_first_hit'] = first[4].name if first[0] and first[4] is not None else 'none'
        panel[key+'_reflected_first_hit'] = second[4].name if second[0] and second[4] is not None else 'none'
        panel[key+'_surface_point'] = [float(v) for v in point]
        panel[key+'_reflected_direction'] = [float(v) for v in reflected]
    front_point = W(0,-0.55,-0.7)
    cabinet['front_receiving_normal_world'] = [float(v) for v in -V]
    for name,pos in [('upper',(-1.67,5.36,1.95)),('left',(-2.85,5.36,-0.08)),('narrow',(0.383,5.36,1.95))]:
        toward_source = (Vector(pos)-front_point).normalized()
        cabinet['front_aperture_cosine_'+name] = float((-V).dot(toward_source))
