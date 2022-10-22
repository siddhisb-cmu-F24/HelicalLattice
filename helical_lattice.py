""" 
MIT License

Copyright (c) 2022 Wen Jiang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

def import_with_auto_install(packages, scope=locals()):
    if isinstance(packages, str): packages=[packages]
    for package in packages:
        if package.find(":")!=-1:
            package_import_name, package_pip_name = package.split(":")
        else:
            package_import_name, package_pip_name = package, package
        try:
            scope[package_import_name] = __import__(package_import_name)
        except ImportError:
            import subprocess
            subprocess.call(f'pip install {package_pip_name}', shell=True)
            scope[package_import_name] =  __import__(package_import_name)
required_packages = "streamlit numpy scipy pandas plotly".split()
import_with_auto_install(required_packages)

from sys import modules
import streamlit as st
import numpy as np

def main():
    title = "HelicalLattice: 2D Lattice ⇔ Helical Lattice"
    st.set_page_config(page_title=title, layout="wide")

    st.title(title)

    if len(st.session_state)<1:  # only run once at the start of the session
        #set_initial_session_state()
        set_session_state_from_query_params()
    
    col1 = st.sidebar
    col2, col3, col4 = st.columns((1, 1, 1), gap="small")

    with col1:
        with st.expander(label="README", expanded=False):
            st.write("**HelicalLattice** is a Web app that helps the user to understand how a helical lattice and its underlying 2D lattice can interconvert. The user can specify any 2D lattice and choose a line segment connecting any pair of lattice points that defines the block of 2D lattice to be rolled up into a helical lattice")

        direction = st.radio(label="Mode:", options=["2D⇒Helical", "Helical⇒2D"], index=0, label_visibility="collapsed", horizontal=True, help="Choose a mode", key="direction")
        if direction == "2D⇒Helical":
            ax = st.number_input('Unit cell vector a.x (Å)', value=50., step=1.0, format="%.2f", help="x coordinate of the unit cell a vector", key="ax")
            ay = st.number_input('Unit cell vector a.y (Å)', value=0., step=1.0, format="%.2f", help="y coordinate of the unit cell a vector", key="ay")
            bx = st.number_input('Unit cell vector b.x (Å)', value=30.0, step=1.0, format="%.2f", help="x coordinate of the unit cell b vector", key="bx")
            by = st.number_input('Unit cell vector b.y (Å)', value=20.0, step=1.0, format="%.2f", help="y coordinate of the unit cell b vector", key="by")
            na = st.number_input('# units along unit cell vector a', value=3, step=1, format="%d", help="# units along unit cell vector a", key="na")
            nb = st.number_input('# units along unit cell vector b', value=0, step=1, format="%d", help="# units along unit cell vector b", key="nb")
        else:
            twist = st.number_input('Twist (°)', value=30.0, min_value=-180., max_value=180., step=1.0, format="%.2f", help="twist", key="twist")
            rise = st.number_input('Rise (Å)', value=20., min_value=0.001, step=1.0, format="%.2f", help="rise", key="rise")
            csym = st.number_input('Axial symmetry', value=3, min_value=1, step=1, format="%d", help="csym", key="csym")
            diameter = st.number_input('Helical diameter (Å)', value=100.0, min_value=0.1, step=1.0, format="%.2f", help="diameter of the helix", key="diameter")

        length = st.number_input('Helical length (Å)', value=400., min_value=0.1, step=1.0, format="%.2f", help="length of the helix", key="length")

        if direction == "Helical⇒2D":
            horizontal = st.checkbox('Set unit cell vector a along x-axis', value=True, help="Set unit cell vector a along x-axis", key="horizontal")
            
        marker_size = st.number_input('Marker size (Å)', value=20., min_value=0.1, step=1.0, format="%.2f", help="size of the markers", key="marker_size")
        figure_height = st.number_input('Plot height (pixels)', value=800, min_value=1, step=10, format="%d", help="height of plots", key="figure_height")

        share_url = st.checkbox('Show sharable URL', value=False, help="Include relevant parameters in the browser URL to allow you to share the URL and reproduce the plots", key="share_url")

        st.markdown("*Developed by the [Jiang Lab@Purdue University](https://jiang.bio.purdue.edu). Report problems to Wen Jiang (jiang12 at purdue.edu)*")

    if direction == "2D⇒Helical":
        a = (ax, ay)
        b = (bx, by)
        twist2, rise2, csym2, diameter2 = convert_2d_lattice_to_helical_lattice(a=a, b=b, endpoint=(na, nb))

        with col2:
            st.subheader("2D Lattice: from which a block of area is selected to be rolled into a helix")
            fig_2d = plot_2d_lattice_plotly(a, b, endpoint=(na, nb), length=length, marker_size=marker_size, figure_height=figure_height)
            st.plotly_chart(fig_2d, use_container_width=True)

        with col3:
            st.subheader("2D Lattice: selected area is ready to be rolled into a helix around the vertical axis")
            fig_helix_unrolled = plot_helical_lattice_unrolled(diameter2, length, twist2, rise2, csym2, marker_size=marker_size, figure_height=figure_height)
            st.plotly_chart(fig_helix_unrolled, use_container_width=True)

        with col4:
            st.subheader("Helical Lattice: rolled up from the selected block of the starting 2D lattice ")
            fig_helix = plot_helical_lattice(diameter2, length, twist2, rise2, csym2, marker_size=marker_size*0.6, figure_height=figure_height)
            st.plotly_chart(fig_helix, use_container_width=True)
    else:
        with col2:
            st.subheader("Helical Lattice")
            fig_helix = plot_helical_lattice(diameter, length, twist, rise, csym, marker_size=marker_size*0.6, figure_height=figure_height)
            st.plotly_chart(fig_helix, use_container_width=True)

        with col3:
            a, b, endpoint = convert_helical_lattice_to_2d_lattice(twist=twist, rise=rise, csym=csym, diameter=diameter,  horizontal=horizontal)

            st.subheader("Helical Lattice: unrolled into a 2D lattice")
            fig_helix_unrolled = plot_helical_lattice_unrolled(diameter, length, twist, rise, csym, marker_size=marker_size, figure_height=figure_height)
            st.plotly_chart(fig_helix_unrolled, use_container_width=True)

        with col4:
            st.subheader("2D Lattice: from which the helix was built")
            fig_2d = plot_2d_lattice_plotly(a, b, endpoint, length=length, marker_size=marker_size, figure_height=figure_height)
            st.plotly_chart(fig_2d, use_container_width=True)

    if share_url:
        set_query_params_from_session_state()
    else:
        st.experimental_set_query_params()

    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False, suppress_st_warning=True)
def plot_2d_lattice_plotly(a=(1, 0), b=(0, 1), endpoint=(10, 0), length=10, marker_size=10, figure_height=500):
  a = np.array(a)
  b = np.array(b)
  na, nb = endpoint
  v0 = na * a + nb * b
  circumference = np.linalg.norm(v0)
  v1 = np.array([-v0[1], v0[0]])
  v1 = length * v1/np.linalg.norm(v1)
  corner_points=[np.array([0, 0]), v0, v0+v1, v1]
  x, y = zip(*(corner_points+[na*a]))
  x0, x1 = min(x), max(x)
  y0, y1 = min(y), max(y)
  xmin = x0-(x1-x0)*0.1
  xmax = x1+(x1-x0)*0.1
  ymin = y0-(y1-y0)*0.1
  ymax = y1+(y1-y0)*0.1

  ia = np.arange(-100, 100)
  ib = np.arange(-100, 100)
  x = []
  y = []
  for j in ib:
    for i in ia:
      v = i*a+j*b
      if xmin <= v[0] <= xmax and ymin <= v[1] <= ymax:
        x.append(v[0])
        y.append(v[1])
  import pandas as pd
  import plotly.express as px
  import plotly.graph_objects as go
  df = pd.DataFrame({'x':x, 'y':y})
  fig = px.scatter(df, x='x', y='y')

  x, y = zip(*corner_points)
  x = [*x, 0]
  y = [*y, 0]
  rectangle = go.Scatter(x=x, y=y, fill="toself", mode='lines', line = dict(color='green', width=marker_size/5, dash='dash'))
  fig.add_trace(rectangle)

  arrow_start = [0, 0]
  arrow_end = na*a
  fig.add_annotation(
    x=arrow_end[0],
    y=arrow_end[1],
    ax= arrow_start[0],
    ay= arrow_start[1],
    xref="x",
    yref="y",
    axref="x",
    ayref="y",
    showarrow=True,
    arrowhead=2,  # type [1,8]
    arrowsize=1,  # relative to arrowwidth
    arrowwidth=3,   # pixel
    arrowcolor="grey",
    opacity=1.0
  )

  arrow_start = na*a
  arrow_end = v0
  fig.add_annotation(
    x=arrow_end[0],
    y=arrow_end[1],
    ax= arrow_start[0],
    ay= arrow_start[1],
    xref="x",
    yref="y",
    axref="x",
    ayref="y",
    showarrow=True,
    arrowhead=2,  # type [1,8]
    arrowsize=1,  # relative to arrowwidth
    arrowwidth=3,   # pixel
    arrowcolor="grey",
    opacity=1.0
  )

  arrow_start = [0, 0]
  arrow_end = v0
  fig.add_annotation(
    x=arrow_end[0],
    y=arrow_end[1],
    ax= arrow_start[0],
    ay= arrow_start[1],
    xref="x",
    yref="y",
    axref="x",
    ayref="y",
    showarrow=True,
    arrowhead=2,  # type [1,8]
    arrowsize=1,  # relative to arrowwidth
    arrowwidth=3,   # pixel
    arrowcolor="red",
    opacity=1.0
  )

  fig.update_traces(marker_size=marker_size, showlegend=False)

  fig.update_layout(
    xaxis =dict(title='X (Å)', range=[xmin, xmax]),
    yaxis =dict(title='Y (Å)', range=[ymin, ymax])
  )
  fig.update_yaxes(scaleanchor = "x", scaleratio = 1)

  #title = "$\\vec{a}=(" + f"{a[0]:.1f}, {a[1]:.1f})Å" + "\\quad\\vec{b}=(" +f"{b[0]:.1f}, {b[1]:.1f})Å" + "\\quad equator=(0,0) \\to" + f"{na}" + "\\vec{a}+" +f"{nb}" + "\\vec{b}$"
  title = f"a=({a[0]:.1f}, {a[1]:.1f})Å\tb=({b[0]:.1f}, {b[1]:.1f})Å\tequator=(0,0)→{na}*a+{nb}*b"
  fig.update_layout(title_text=title, title_x=0.5)
  fig.update_layout(height=figure_height, width=1.2*figure_height/((ymax-ymin)/(xmax-xmin)), autosize=False)
  fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')

  return fig


@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False, suppress_st_warning=True)
def plot_helical_lattice_unrolled(diameter, length, twist, rise, csym, marker_size=10, figure_height=800):
  circumference = np.pi*diameter
  if rise>0:
    n = min(int(length/2/rise)+2, 1000)
    i = np.arange(-n, n+1)
    xs = []
    ys = []
    syms = []
    for si in range(csym):
      x = np.fmod(twist * i + si/csym*360, 360)
      x[x> 360] -= 360
      x[x< 0] += 360
      y = rise * i
      xs.append(x)
      ys.append(y)
      syms.append(np.array([si]*len(x)))
  x = np.concatenate(xs)
  y = np.concatenate(ys)
  sym = np.concatenate(syms)

  import pandas as pd
  df = pd.DataFrame({'x':x, 'y':y, 'csym':sym})
  df["csym"] = df["csym"].astype(str)
  
  import plotly.express as px
  import plotly.graph_objects as go
  fig = px.scatter(df, x='x', y='y', color='csym' if csym>1 else None)

  if twist>=0:
    arrow_start = [0, 0]
    arrow_end = [twist, rise]
  else:
    arrow_start = [360, 0]
    arrow_end = [360+twist, rise]
  fig.add_annotation(
    x=arrow_end[0],
    y=arrow_end[1],
    ax= arrow_start[0],
    ay= arrow_start[1],
    xref="x",
    yref="y",
    axref="x",
    ayref="y",
    showarrow=True,
    arrowhead=2,  # type [1,8]
    arrowsize=1,  # relative to arrowwidth
    arrowwidth=4,   # pixel
    arrowcolor="red",
    opacity=1.0
  )

  i = np.arange(-n, n+1, 0.01)
  for si in range(csym):
    x = np.fmod(twist * i + si/csym*360, 360)
    x[x> 360] -= 360
    x[x< 0] += 360
    y = rise * i
    color = fig.data[si].marker.color
    line = go.Scatter(x=x, y=y, mode ='lines', line = dict(color=color, width=marker_size/10, dash='dot'), opacity=1, showlegend=False)
    fig.add_trace(line)
  equator = go.Scatter(x=[0,360], y=[0,0], xaxis='x', line = dict(color='grey', width=marker_size/3, dash='dash'))
  fig.add_trace(equator)
  fig.update_traces(marker_size=marker_size, showlegend=False)

  fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 360/circumference
  )
  fig.update_layout(
    xaxis =dict(title='twist (°)', range=[0,360], tickvals=np.linspace(0,360,13)),
    yaxis =dict(title='rise (Å)', range=[-length/2, length/2]),
  )
  
  title = f"pitch={rise*abs(360/twist):.1f}Å\ttwist={twist:.1f}° rise={rise:.1f}Å sym=c{csym} diameter={diameter:.1f}Å circumference={circumference:.1f}Å"
  fig.update_layout(title_text=title, title_x=0.5)
  fig.update_layout(height=figure_height, width=1.2*figure_height*circumference/length, autosize=False)
  fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')

  return fig

@st.experimental_memo(persist='disk', max_entries=1, show_spinner=False, suppress_st_warning=True)
def plot_helical_lattice(diameter, length, twist, rise, csym,  marker_size = 10, figure_height=500):
  if rise>0:
    n = min(int(length/2/rise)+2, 1000)
    i = np.arange(-n, n+1)
    xs = []
    ys = []
    zs = []
    syms = []
    for si in range(csym):
      x = diameter/2 * np.cos(np.deg2rad(twist)*i+si/csym*2*np.pi)
      y = diameter/2 * np.sin(np.deg2rad(twist)*i+si/csym*2*np.pi)
      z = i * rise
      xs.append(x)
      ys.append(y)
      zs.append(z)
      syms.append(np.array([si]*len(z)))
  x = np.concatenate(xs)
  y = np.concatenate(ys)
  z = np.concatenate(zs)
  sym = np.concatenate(syms)

  import pandas as pd
  df = pd.DataFrame({'x':x, 'y':y, 'z':z, 'csym':sym})
  df["csym"] = df["csym"].astype(str)
  
  import plotly.express as px
  import plotly.graph_objects as go
  fig = px.scatter_3d(df, x='x', y='y', z='z', labels={'x': 'X (Å)', 'y':'Y (Å)', 'z':'Z (Å)'}, color='csym' if csym>1 else None)
  fig.update_traces(marker_size = marker_size)

  i = np.arange(-n, n+1, 0.1)
  for si in range(csym):
    x = diameter/2 * np.cos(np.deg2rad(twist)*i+si/csym*2*np.pi)
    y = diameter/2 * np.sin(np.deg2rad(twist)*i+si/csym*2*np.pi)
    z = i * rise
    color = fig.data[si].marker.color
    spiral = go.Scatter3d(x=x, y=y, z=z, mode ='lines', line = dict(color=color, width=marker_size/2), opacity=1, showlegend=False)
    fig.add_trace(spiral)

  def cylinder(r, h, z0=0, n_points=100, nv =50):
    theta = np.linspace(0, 2*np.pi, n_points)
    v = np.linspace(z0, z0+h, nv )
    theta, v = np.meshgrid(theta, v)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = v
    return x, y, z
  
  def equator_circle(r, z, n_points=36):
    theta = np.linspace(0, 2*np.pi, n_points)
    x= r*np.cos(theta)
    y = r*np.sin(theta)
    z0 = z*np.ones(theta.shape)
    return x, y, z0

  x, y, z = cylinder(r=diameter/2-marker_size/2, h=length, z0=-length/2)
  colorscale = [[0, 'white'], [1, 'white']]
  cyl = go.Surface(x=x, y=y, z=z, colorscale = colorscale, showscale=False, opacity=0.8)
  fig.add_trace(cyl)
  x, y, z = equator_circle(r=diameter/2, z=0)
  equator = go.Scatter3d(x=x, y=y, z=z, mode ='lines', line = dict(color='grey', width=marker_size/2, dash='dash'), opacity=1, showlegend=False)
  fig.add_trace(equator)

  title = f"pitch={rise*abs(360/twist):.1f}Å\ttwist={twist:.1f}° rise={rise:.1f}Å sym=c{csym} diameter={diameter:.1f}Å circumference={np.pi*diameter:.1f}Å"
  fig.update_layout(title_text=title, title_x=0.5)

  camera = dict(
    up=dict(x=0, y=0, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0., y=1, z=0)
  )
  fig.update_layout(scene_camera=camera)

  fig.update_scenes(zaxis=dict(range=[-length/2,length/2]), xaxis_visible=False, yaxis_visible=False, zaxis_visible=False, camera_projection_type='orthographic', aspectmode='data')
  fig.update_layout(height=figure_height, autosize=True)
  fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)')

  return fig


@st.experimental_memo(max_entries=10, show_spinner=False, suppress_st_warning=True)
def convert_2d_lattice_to_helical_lattice(a=(1, 0), b=(0, 1), endpoint=(10, 0)):
  def set_to_periodic_range(v, min=-180, max=180):
    from math import fmod
    tmp = fmod(v-min, max-min)
    if tmp>=0: tmp+=min
    else: tmp+=max
    return tmp
  def length(v):
    return np.linalg.norm(v)
  def transform_vector(v, vref=(1, 0)):
    ang = np.arctan2(vref[1], vref[0])
    cos = np.cos(ang)
    sin = np.sin(ang)
    m = [[cos, sin], [-sin, cos]]
    v2 = np.dot(m, v.T)
    return v2
  def on_equator(v, epsilon=0.5):
      # test if b vector is on the equator
      if abs(v[1]) > epsilon: return 0
      return 1
  
  a, b, endpoint = map(np.array, (a, b, endpoint))
  na, nb = endpoint
  v_equator = na*a + nb*b
  circumference = length(v_equator)
  va = transform_vector(a, v_equator)
  vb = transform_vector(b, v_equator)
  minLength = max(1.0, min(np.linalg.norm(va), np.linalg.norm(vb)) * 0.9)
  vs_on_equator = []
  vs_off_equator = []
  epsilon = 0.5
  maxI = 10
  for i in range(-maxI, maxI + 1):
      for j in range(-maxI, maxI + 1):
          if i or j:
              v = i * va + j * vb
              v[0] = set_to_periodic_range(v[0], min=0, max=circumference)
              if np.linalg.norm(v) > minLength:
                  if v[1]<0: v *= -1
                  if on_equator(v, epsilon=epsilon):
                      vs_on_equator.append(v)
                  else:
                      vs_off_equator.append(v)
  twist, rise, csym = 0, 0, 1
  if vs_on_equator:
      vs_on_equator.sort(key=lambda v: abs(v[0]))
      best_spacing = abs(vs_on_equator[0][0])
      csym_f = circumference / best_spacing
      expected_spacing = circumference/round(csym_f)
      if abs(best_spacing - expected_spacing)/expected_spacing < 0.05:
          csym = int(round(csym_f))
  if vs_off_equator:
      vs_off_equator.sort(key=lambda v: (abs(round(v[1]/epsilon)), abs(v[0])))
      twist, rise = vs_off_equator[0]
      twist *= 360/circumference
      if csym>1 and abs(twist)>180./csym:
          if twist<0: twist+=360./csym
          elif twist>0: twist-=360./csym
  diameter = circumference/np.pi
  return twist, rise, csym, diameter

@st.experimental_memo(max_entries=10, show_spinner=False, suppress_st_warning=True)
def convert_helical_lattice_to_2d_lattice(twist=30, rise=20, csym=1, diameter=100, horizontal=True):
  def angle(v1, v2):  # angle between two vectors, ignoring vector polarity
      p = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
      p = np.clip(abs(p), 0, 1)
      ret = np.rad2deg(np.arccos(p))  # 0<=angle<90
      return ret
  def transform_vector(v, vref=(1, 0)):
    ang = np.arctan2(vref[1], vref[0])
    cos = np.cos(ang)
    sin = np.sin(ang)
    m = [[cos, sin], [-sin, cos]]
    v2 = np.dot(m, v.T)
    return v2
  
  n = np.tile(np.arange(5*360/abs(twist)), reps=(2,1)).T
  v = np.array([twist, rise], dtype=float) * n
  if csym>1:
    vs = []
    for ci in range(csym):
      tmp = v * 1.0
      tmp[:, 0] += ci/csym * 360
      vs.append(tmp)
    v = np.vstack(vs)
  v[:, 0] = np.fmod(v[:, 0], 360)
  v[v<0] += 360
  v[:, 0] *= np.pi*diameter/360 # convert x-axis values from angles to distances
  dist = np.linalg.norm(v, axis=1)
  dist_indices = np.argsort(dist)

  v = v[dist_indices] # now sorted from short to long distance
  err = 1.0 # max angle between 2 vectors to consider non-parallel
  va = v[1]
  for i in range(1, len(v)):
    if angle(va, v[i])> err:
      vb = v[i]
      break
  va, vb = vb, va # set va to be the longer unit cell vector

  ve = np.array([np.pi*diameter, 0])
  dist = []
  for ia in range(-100, 100):
    for ib in range(-100, 100):
      vtmp = ia*va + ib*vb
      err = np.linalg.norm(ve-vtmp)
      dist.append((err, ia, ib))
  dist.sort()
  endpoint = dist[0][1:]

  if horizontal:
    vb = transform_vector(vb, vref=va)
    va = np.array([np.linalg.norm(va), 0.0])
  return va, vb, endpoint

int_types = {'csym':3, 'figure_height':800, 'horizontal':1, 'na':3, 'nb':0, 'share_url':0}
float_types = {'ax':50.0, 'ay':0.0, 'bx':30.0, 'by':20.0, 'diameter':200.0, 'length':400.0, 'marker_size':20, 'rise':20.0, 'twist':30.0}
default_values = int_types | float_types | {'direction':'2D⇒Helical', }
def set_initial_session_state():
    for attr in sorted(default_values.keys()):
            if attr in int_types:
                st.session_state[attr] = int(default_values[attr])
            elif attr in float_types:
                st.session_state[attr] = float(default_values[attr])
            else:
                st.session_state[attr] = default_values[attr]

def set_query_params_from_session_state():
    d = {}
    attrs = sorted(st.session_state.keys())
    for attr in attrs:
        v = st.session_state[attr]
        if attr in default_values and v==default_values[attr]: continue
        if attr in int_types or isinstance(v, bool):
            d[attr] = int(v)
        elif attr in float_types:
            d[attr] = f'{float(v):g}'
        else:
            d[attr] = v
    st.experimental_set_query_params(**d)

def set_session_state_from_query_params():
    query_params = st.experimental_get_query_params()
    for attr in sorted(query_params.keys()):
            if attr in int_types:
                st.session_state[attr] = int(query_params[attr][0])
            elif attr in float_types:
                st.session_state[attr] = float(query_params[attr][0])
            else:
                st.session_state[attr] = query_params[attr][0]

@st.experimental_memo(persist='disk', show_spinner=False)
def setup_anonymous_usage_tracking():
    try:
        import pathlib, stat
        index_file = pathlib.Path(st.__file__).parent / "static/index.html"
        index_file.chmod(stat.S_IRUSR|stat.S_IWUSR|stat.S_IRGRP|stat.S_IROTH)
        txt = index_file.read_text()
        if txt.find("gtag/js?")==-1:
            txt = txt.replace("<head>", '''<head><script async src="https://www.googletagmanager.com/gtag/js?id=G-CTBKF6J4CG"></script><script>window.dataLayer = window.dataLayer || [];function gtag(){dataLayer.push(arguments);}gtag('js', new Date());gtag('config', 'G-CTBKF6J4CG');</script>''')
            index_file.write_text(txt)
    except:
        pass

if __name__ == "__main__":
    setup_anonymous_usage_tracking()
    main()
