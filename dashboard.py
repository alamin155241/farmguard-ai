"""FarmGuard-AI monitoring, physical-trial collection, and ML dashboard."""
from __future__ import annotations
import json, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
from audit_dataset import audit_dataset
from camera_service import CameraVerifier
from dataset_service import DatasetStore, ML_FIELDS, aggregate_trial, quality_warnings
from ml_inference import model_status, predict_trial
from live_test_service import persist_completed_live_test
from mqtt_service import BROKER, PORT, TOPIC, MQTTService
from protocol_60_20 import MANIFEST_PATH, SPLIT_METADATA_PATH
from train_model import train_development

st.set_page_config(page_title="FarmGuard-AI", page_icon="🌾", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.2rem;max-width:1400px}.state-card{border-radius:16px;padding:22px;text-align:center;color:white;font-size:2.4rem;font-weight:800;letter-spacing:.08em}.NORMAL{background:linear-gradient(135deg,#15803d,#22c55e)}.WATCH{background:linear-gradient(135deg,#b45309,#f59e0b)}.CONFIRMED{background:linear-gradient(135deg,#991b1b,#ef4444);animation:pulse 1.2s infinite}.UNKNOWN{background:#64748b}.alarm{background:#7f1d1d;color:white;border:3px solid #fca5a5;border-radius:16px;padding:22px;text-align:center;font-size:2rem;font-weight:900;margin:14px 0}@keyframes pulse{50%{filter:brightness(1.25)}}div.stButton>button{min-height:3rem;font-weight:700}
</style>""",unsafe_allow_html=True)

@st.cache_resource
def resources():
    mqtt=MQTTService(); mqtt.start(); return mqtt,DatasetStore(),CameraVerifier()
service,store,camera=resources()

def dataset_frame():
    try: return pd.read_csv(store.path)
    except (FileNotFoundError,pd.errors.EmptyDataError): return pd.DataFrame(columns=ML_FIELDS)

@st.fragment(run_every="1s")
def monitor():
    snap=service.snapshot(); latest=snap["latest"] or {}; state=latest.get("state","UNKNOWN")
    c1,c2,c3=st.columns(3); c1.metric("MQTT Connection","● Connected" if snap["connected"] else "○ Disconnected"); c2.metric("Broker",f"{BROKER}:{PORT}"); c3.metric("Last Message",snap["last_message"] or "Waiting for data…")
    st.caption(f"Subscribed to `{TOPIC}`"+(f" · {snap['device_status']}" if snap["device_status"] else ""))
    st.markdown(f'<div class="state-card {state}">{state}</div>',unsafe_allow_html=True)
    if state=="CONFIRMED": st.markdown('<div class="alarm">🚨 INTRUSION CONFIRMED 🚨</div>',unsafe_allow_html=True)
    cols=st.columns(5); cols[0].metric("PIR Motion","DETECTED" if latest.get("pir") else "CLEAR"); cols[1].metric("LDR Raw Value",latest.get("ldr","—")); cols[2].metric("Laser Beam","BLOCKED" if latest.get("beam_blocked") else ("CLEAR" if latest else "—")); cols[3].metric("MQ Gas Raw Value",latest.get("gas","—")); temp=latest.get("temperature"); cols[4].metric("Temperature",f"{temp:.2f} °C" if temp is not None else "—")
    history=snap["history"][-100:]
    ml_name="MODEL NOT TRAINED"; ml_probability=None
    if model_status() and history:
        try:
            live_row=aggregate_trial(history[-16:],0,8.0)
            ml_pred,ml_probability=predict_trial(live_row); ml_name="INTRUSION" if ml_pred else "NORMAL"
        except (ValueError,FileNotFoundError,KeyError): pass
    st.subheader("Parallel decision & visual verification")
    auto_camera=st.toggle("Auto-check webcam on WATCH / CONFIRMED",value=True,key="auto_camera")
    manual=st.button("CHECK WEBCAM NOW",use_container_width=True,key="camera_now")
    if manual or (auto_camera and camera.should_auto_verify(state)): camera.start()
    camera_snap=camera.snapshot(); camera_result=camera_snap["result"]; camera_name="VERIFYING…" if camera_snap["running"] else "IDLE"
    if camera_snap["person_detected"]: camera_name="PERSON DETECTED"
    elif camera_result: camera_name="ERROR" if camera_result.error else "NO PERSON"
    final="VERIFIED INTRUSION" if camera_snap["person_detected"] and (state=="CONFIRMED" or ml_name=="INTRUSION") else ("ALERT — AWAITING VISUAL VERIFICATION" if state=="CONFIRMED" else "NO VERIFIED INTRUSION")
    decisions=st.columns(4); decisions[0].metric("Rule-based state",state); decisions[1].metric("Sensor ML",ml_name, f"{ml_probability:.1%} intrusion" if ml_probability is not None else None); decisions[2].metric("Camera",camera_name); decisions[3].metric("FINAL",final)
    if camera_snap["error"] and not camera_snap["running"]: st.warning(camera_snap["error"])
    if camera_snap["frame"] is not None: st.image(camera_snap["frame"],channels="BGR",caption=f"Live verification · {camera_snap['detector']} · {camera_snap['frames_processed']} frames",width=520)
    if history:
        frame=pd.DataFrame(history); frame["time"]=pd.to_datetime(frame["timestamp"]); charts=st.columns(3)
        for col,field,title in zip(charts,["ldr","gas","temperature"],["LDR","Gas","Temperature °C"]):
            with col: st.caption(title); st.line_chart(frame.set_index("time")[[field]],height=200)
        display=frame[["timestamp","pir","ldr","beam_blocked","gas","temperature","state"]].iloc[::-1]; display.columns=["Timestamp","PIR","LDR","Beam Blocked","Gas","Temperature","State"]; st.dataframe(display,use_container_width=True,hide_index=True,height=300)
    else: st.info("Waiting for the first FarmGuard sensor reading…")
    if snap["last_error"]: st.caption(f"Subscriber note: {snap['last_error']}")

def start_trial(prefix,label):
    st.session_state[f"{prefix}_trial"]={"phase":"countdown","label":label,"started":time.time()}

@st.fragment(run_every="500ms")
def trial_panel(prefix="collect",save=True):
    key=f"{prefix}_trial"; trial=st.session_state.get(key)
    if not trial:
        a,b=st.columns(2)
        a.button("START NORMAL TRIAL",key=f"{prefix}_normal",use_container_width=True,on_click=start_trial,args=(prefix,0))
        b.button("START INTRUSION TRIAL",key=f"{prefix}_intrusion",use_container_width=True,on_click=start_trial,args=(prefix,1))
        return
    elapsed=time.time()-trial["started"]; label=trial["label"]; name="NORMAL" if label==0 else "INTRUSION"
    st.metric("Current trial type",name)
    if elapsed<3:
        st.warning(f"Prepare… {max(1,3-int(elapsed))}"); st.progress(elapsed/11); return
    if trial["phase"]=="countdown":
        trial["phase"]="recording"; trial["marker"]=service.snapshot()["sequence"]; trial["record_started"]=time.time()
        if prefix=="live": camera.start()
    capture=min(8.0,time.time()-trial["record_started"]); samples=service.samples_after(trial["marker"])
    st.success("START NOW — RECORDING"); st.progress(min(1.0,capture/8)); st.metric("MQTT messages captured",len(samples)); st.caption(f"{capture:.1f} / 8.0 seconds")
    if prefix=="live":
        camera_snap=camera.snapshot()
        st.caption(f"Camera verification: {'RUNNING' if camera_snap['running'] else 'COMPLETE'} · {camera_snap['frames_processed']} frames · {camera_snap['detector']}")
        if camera_snap["frame"] is not None: st.image(camera_snap["frame"],channels="BGR",caption="Live camera verification",width=520)
    if capture<8: return
    st.session_state.pop(key,None)
    try:
        row=aggregate_trial(samples,label,8.0); warnings=quality_warnings(row)
        if save: store.append(row); st.session_state["last_saved_row"]=row
        else:
            pred,prob=predict_trial(row); states=[s.get("state") for s in samples]; rule="CONFIRMED" if "CONFIRMED" in states else ("WATCH" if "WATCH" in states else "NORMAL")
            st.session_state["pending_live_test_result"]={"row":row,"truth":name,"rule":rule,"prediction":"INTRUSION" if pred else "NORMAL","probability":prob,"correct":pred==label,"warnings":warnings}
    except (ValueError,FileNotFoundError) as exc: st.session_state[f"{prefix}_error"]=str(exc)
    # Completion must rerun the full app: the persistent LIVE ML TEST result
    # renderer lives outside this fragment.
    st.rerun()

st.title("🌾 FarmGuard-AI Command Center")
monitor_tab,collect_tab,model_tab,live_tab=st.tabs(["Live Monitoring","ML Dataset Collection","ML Model","LIVE ML TEST"])
with monitor_tab: monitor()
with collect_tab:
    st.header("Frozen Offline Dataset")
    st.info("The original 80 trials are frozen under the post-exploratory 60/20 protocol. Collection is locked to prevent accidental mixing with either offline split.")
    if Path(MANIFEST_PATH).exists():
        manifest=pd.read_csv(MANIFEST_PATH); dev=manifest[manifest["split"]=="development"]; final=manifest[manifest["split"]=="final_test"]
        cols=st.columns(4)
        cols[0].metric("Development NORMAL",int((dev["label"]==0).sum()))
        cols[1].metric("Development INTRUSION",int((dev["label"]==1).sum()))
        cols[2].metric("Final-test NORMAL",int((final["label"]==0).sum()))
        cols[3].metric("Final-test INTRUSION",int((final["label"]==1).sum()))
        st.caption("Development: 60 trials / Final test: 20 trials / LIVE ML TEST: separate")
    else: st.error("Frozen split manifest is missing; collection remains locked.")
    st.button("DATASET COLLECTION LOCKED",disabled=True,use_container_width=True)
    if st.session_state.get("collect_error"): st.error(st.session_state.pop("collect_error"))
    if st.session_state.get("last_saved_row"):
        row=st.session_state["last_saved_row"]; st.success("Trial saved — exactly one dataset row appended."); [st.warning(w) for w in quality_warnings(row)]; st.dataframe(pd.DataFrame([row]),hide_index=True,use_container_width=True)
    df=dataset_frame()
    if len(df): st.download_button("Download ML dataset CSV",store.path.read_bytes(),"ml_dataset.csv","text/csv"); st.dataframe(df.iloc[::-1],hide_index=True,use_container_width=True,height=320)
with model_tab:
    st.header("60-Development / 20-Final-Test Protocol")
    df=dataset_frame(); counts=df["label"].value_counts().to_dict() if len(df) else {}; meta=model_status(); a,b,c=st.columns(3); a.metric("Frozen source",len(df)); b.metric("Development","60 (30 / 30)"); c.metric("Development model","Trained" if Path("models/decision_tree_dev60.joblib").exists() else "Not trained")
    x,y,z=st.columns(3)
    if x.button("Audit Dataset",use_container_width=True):
        try: st.session_state["audit_report"]=audit_dataset()
        except Exception as exc: st.session_state["audit_report"]={"error":str(exc)}
    enough=Path(SPLIT_METADATA_PATH).exists()
    if y.button("Train Development Model",disabled=not enough,use_container_width=True):
        try: st.session_state["train_report"]=train_development()
        except Exception as exc: st.session_state["train_report"]={"error":str(exc)}
    z.button("Refresh Results",use_container_width=True)
    if not enough: st.warning("Frozen 60/20 protocol metadata is missing.")
    else: st.warning("Training and five-fold CV use only 60 development trials. The locked 20-row final evaluation has not been executed.")
    if st.session_state.get("audit_report"): st.json(st.session_state["audit_report"])
    development_metrics_path=Path("results/development_cv_metrics.json")
    if development_metrics_path.exists():
        development_metrics=json.loads(development_metrics_path.read_text(encoding="utf-8")); selected=development_metrics["development_oof_metrics"]
        st.caption(f"Last development training: {(model_status() or {}).get('training_timestamp','unknown')} / Final test: LOCKED - NOT EXECUTED")
        cols=st.columns(4)
        for col,k in zip(cols,["accuracy","precision","recall","f1"]): col.metric("Development OOF "+k.title(),f"{selected[k]:.3f}")
        st.json({"selected_parameters":development_metrics["selected_parameters"],"fold_f1":development_metrics["selected_fold_f1"],"cv_f1_mean":development_metrics["cv_f1_mean"],"cv_f1_std":development_metrics["cv_f1_std"]})
    metrics_path=Path("results/legacy_metrics_display_disabled.json")
    if metrics_path.exists():
        metrics=json.loads(metrics_path.read_text(encoding="utf-8")); selected=metrics["evaluation"][metrics["chosen_model"]]; st.caption(f"Last training: {(model_status() or {}).get('training_timestamp','—')} · Held-out test samples: {metrics['held_out_test_samples']}"); cols=st.columns(4)
        for col,k in zip(cols,["accuracy","precision","recall","f1"]): col.metric(k.title(),f"{selected[k]:.3f}")
        imgs=st.columns(3)
        for col,path,title in zip(imgs,["results/confusion_matrix.png","results/feature_importance.png","results/decision_tree.png"],["Confusion matrix","Feature importance","Decision Tree"]):
            if Path(path).exists(): col.image(path,caption=title,use_container_width=True)
        if Path("results/baseline_comparison.csv").exists(): st.subheader("Baseline comparison"); st.dataframe(pd.read_csv("results/baseline_comparison.csv"),hide_index=True,use_container_width=True)
        if Path("results/model_rules.txt").exists():
            with st.expander("Decision Tree learned rules"):
                st.code(Path("results/model_rules.txt").read_text(encoding="utf-8"),language="text")
with live_tab:
    st.header("LIVE ML TEST"); st.info("Select the true physical action, then perform it during the 8-second recording. This test is not added to training data.")
    if not model_status(): st.warning("Train a model before using live ML test.")
    else:
        trial_panel("live",False)
        if st.session_state.get("live_error"): st.error(st.session_state.pop("live_error"))
        pending=st.session_state.get("pending_live_test_result")
        if pending:
            camera_snap=camera.snapshot()
            completed=persist_completed_live_test(st.session_state,pending,camera_snap)
            if completed is not None: st.session_state.pop("pending_live_test_result",None)
            else: st.info("Waiting for sensor/ML and camera verification to both finish…")
        result=st.session_state.get("last_live_test_result")
        if result:
            st.subheader("Completed LIVE ML TEST")
            probability_text=f"{result['ml_probability']:.1%}" if result["ml_probability"] is not None else "Unavailable"
            st.markdown(
                f"**Ground Truth : {result['ground_truth']}**  \n"
                f"**Rule State : {result['rule_state']}**  \n"
                f"**Sensor ML : {result['ml_prediction']}**  \n"
                f"**ML Probability : {probability_text}**  \n"
                f"**Camera : {result['camera_result']}**  \n"
                f"**Correctness : {result['correctness']}**  \n"
                f"**FINAL : {result['final_result']}**"
            )
            if result.get("camera_error"): st.warning(result["camera_error"])
            if result.get("camera_frame") is not None: st.image(result["camera_frame"],channels="BGR",caption=f"{result['camera_detector']} · {result['camera_frames_processed']} frames",width=520)
            [st.warning(w) for w in result.get("warnings",[])]
with st.sidebar: st.caption(f"Python: `{sys.executable}`"); st.caption("ML labels come only from physical trial buttons.")
