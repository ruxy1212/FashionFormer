import streamlit as st
from local_components import card_container, card_container_border, ch1, ch2, ch3, ch4
from streamlit_extras.tags import tagger_component
import io
import cv2
import random
import numpy as np
from PIL import Image
from argparse import ArgumentParser

all_colors = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'gray', 'lightblue', 'maroon', 'navy', 'purple', 'fuchsia', 'olive', 'teal', 'aqua', 'cyan', 'tomato', 'lime', 'orange', 'gold']

if "model_init" not in st.session_state.keys() and "model_det" not in st.session_state.keys():
    st.session_state["model_init"] = 'model_initialized' #loadPretrainedModel(embed_size, loss_type)
    from mmdet.apis import  inference_detector,init_detector
    st.session_state["model_det"] = inference_detector,init_detector
else:
    inference_detector,init_detector = st.session_state["model_det"]
# model = st.session_state["model"]
print(st.session_state["model_init"])


#loading configs and weights
def load_model():
    config_file='configs/fashionformer/fashionpedia/fashionformer_r101_mlvl_feat_8x.py'
    checkpoint_file='../data/model_final.pth'
    model=init_detector(config_file, checkpoint_file, device="cpu")
    return model

def search_products(out_det):
    if out_det is not None:
        w_factor = [5, 2, 1, 0.5] #category,supercategory,color,attributes
        for obj in out_det:
            obj_cls = obj[0]
            obj_attr = obj[1]
            obj_clr = obj[2]
            st.write(obj_cls)
            st.write(obj_attr)
            st.write(obj_clr)
def infer_model(model, img):
    result = inference_detector(model, img)
    ret = model._show_result(img, result, score_thr=0.5, out_file=None)
    # out_img, out_det = ret
    return ret
    # st.image(out_img)
    # for obj in out_det:
    #     obj_cls = obj[0]
    #     obj_attr = obj[1]
    #     obj_clr = obj[2]
    #     st.write(obj_cls)
    #     st.write(obj_attr)
    #     st.write(obj_clr)
    # st.write("📍 The user specifies a safe zone by drawing a polygon and the violations are detected accordingly.")
    # st.write("📍 This can be used in factories to identify safe zones and make sure workers are walking within them.")
    # st.write("📍 This can also be used in zebra crossings to ensure pedestrian safety.")
    #  st.image(img)
    # PYTHONPATH='.' python demo/ntt.py demo/girl.png configs/fashionformer/fashionpedia/fashionformer_r101_mlvl_feat_8x.py ../data/model_final.pth --device cpu --out-file demoe.png --score-thr 0.5
def main():
    if "model_saved" not in st.session_state.keys():
        model=load_model()
        st.session_state['model_saved'] = model
    else:
        model = st.session_state['model_saved']
    with st.container():
        st.header("Image Segmentation App")
        # st.sidebar.markdown("1. People")
        # st.title("No Walk Zone Violation 🧍")
        # st.write("📍 The user specifies a safe zone by drawing a polygon and the violations are detected accordingly.")
        # st.write("📍 This can be used in factories to identify safe zones and make sure workers are walking within them.")
        # st.write("📍 This can also be used in zebra crossings to ensure pedestrian safety.")
        # drawing_mode = st.sidebar.selectbox("Drawing tool:", ("polygon","line", "rect", "circle","point"))
        st.subheader("Dashboard")
        file_upload = st.file_uploader("Upload Image:", type=["png", "jpg", "jpeg"])

        if file_upload is not None and model:
            img = np.asarray(Image.open(io.BytesIO(file_upload.getvalue())))
            cv_img = 'image.jpg'
            # img = cv2.imdecode(np.asarray(bytearray(file_upload.getvalue()), np.uint8), cv2.IMREAD_UNCHANGED) #bytearray(image.read())
            # if img.shape[-1] == 4:  # Check if image has alpha channel
            #     img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            # else:
            #     img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR) 
            img = cv2.imdecode(np.frombuffer(file_upload.getvalue(), np.uint8), cv2.IMREAD_COLOR)  # Use IMREAD_COLOR for color images
            if img.shape[-1] == 4:  # Check for alpha channel
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            st.image(img, channels="BGR")  # Specify BGR channel order for display
            # st.image(img)
            # p_img = Image.open(io.BytesIO(file_upload.getvalue()))
            # p_img.save(cv_img)
            # st.image(np.asarray(p_img))
            st.write("Processing image...")
            out_img, out_det = infer_model(model, img)
            st.image(out_img, channels="BGR")
            # st.image(out_img, channels="RGB")
            for k, itm in enumerate(out_det):
                det = itm[2].split('\n')
                titl = itm[0].replace('|', ' - ')
                with card_container(key=f"card{k}"):
                    ch4(text=f"Category: {itm[1]}  |  Color: {itm[3]}", weight="normal")
                    ch2(text=titl, weight="bolder") 
                    tagger_component("Attributes: ", det[:-1], color_name= [random.choice(all_colors) for _ in det[:-1]])
                        
            # cols = st.columns(3)
            # Determine number of columns based on screen size
            # num_cols = st.columns.width
            # cols = 4
            # if num_cols < 1200:
            #     cols = 3  # Adjust breakpoints for lg screens as needed
            # if num_cols < 700:
            #     cols = 2  # Adjust breakpoints for md screens as needed
            # if num_cols < 500:
            #     cols = 1  # Adjust breakpoints for sm screens as needed
            # for i in range(0, len(out_det), cols):
            #     row_cards = out_det[i:i + cols]
            #     inner_cols = st.columns(cols)
            #     for j in range(0, inner_cols):
            #         k = (i*cols)+j
            #         with inner_cols[j]:
            #             itm = out_det[k]
            #             det = itm[2].split('\n')
            #             dets = ', '.join(['#' + z for z in det])
            #             titl = itm[0].replace('|', ' - ')
            #             with card_container(key=f"card{k}"):
            #                 ch3(text=f"Category: {itm[1]}    Color: {itm[3]}", weight="normal")
            #                 ch1(text=titl, weight="bolder") 
            #                 tagger_component("Attributes: ", det, color_name= [random.choice(all_colors) for _ in det])
                            

                        # b_types = ['default', 'secondary', 'outline', 'destructive']
                        # dets = [(z, random.choice(b_types)) for z in det]
                        # with ui.card(key="card1"):
                            # ui.element("span", children=["Email"], className="text-gray-400 text-sm font-medium m-1", key="label1")
                            # ui.element("input", key="email_input", placeholder="Your email")

                            # ui.element("span", children=["User Name"], className="text-gray-400 text-sm font-medium m-1", key="label2")
                            # ui.element("input", key="username_input", placeholder="Create a User Name")
                            # ui.element("button", text="Submit", key="button", className="m-1")
                        # ui.card(title=f"Category: {itm[0]}", content=f"Type: {itm[1]}    Color: {itm[3]}", description=dets).render()


                
            # with cols[0]:
            # with ui.card():
            #     ui.element()
            #     ui.card(title="Total Revenue", content="$45,231.89", description="+20.1% from last month", key="card1").render()
            # with cols[1]:
            #     ui.card(title="Subscriptions", content="+2350", description="+180.1% from last month", key="card2").render()
            # with cols[2]:
            #     ui.card(title="Sales", content="+12,234", description="+19% from last month", key="card3").render()

            # for obj in out_det:
            #     obj_cls = obj[0]
            #     obj_attr = obj[2]
            #     obj_clr = obj[3]
            #     obj_spr = obj[1]
            #     st.write(obj_cls+"("+obj_spr+")")
            #     st.write(obj_attr)
            #     st.write(obj_clr)
            # st.image(img)
            if st.button("Search for Products") and model:
                col1, col2 = st.columns(2)
                with col1:
                    with card_container_border(key=f"card1"):
                        st.image(img, channels="BGR")
                    st.write("col1")
                    st.form_submit_button("Button 1")
                with col2:
                    st.write("col2")
                    st.form_submit_button("Button 2")
                # search_products(out_det)
                st.write("📍 The user specifies a safe zone by drawing a polygon and the violations are detected accordingly.")
                st.write("📍 This can be used in factories to identify safe zones and make sure workers are walking within them.")
                st.write("📍 This can also be used in zebra crossings to ensure pedestrian safety.")
        # image,list_coor=draw_bound(drawing_mode,bg_image)
        # result=st.button('Search for Products')

        # if result and model and image:
            # st.write("Calculating results...")
            # infer_model(model, image)
        # else:
            # st.write("Upload an image first")


if __name__=='__main__':
    main()