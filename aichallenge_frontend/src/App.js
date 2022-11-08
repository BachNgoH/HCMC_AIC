import React, { useContext } from "react";
import TextQueryForm from "./components/query/TextQueryForm";
import classes from "./App.module.css";
import Logo from "./components/query/Logo";
import ImageList from "./components/search/ImageList";
import { useState } from "react";
import VideoModal from "./components/search/VideoModal";
import "../node_modules/rsuite/dist/rsuite.min.css";
import SubmissionList from "./components/submissions/SubmissionList";
import SubmitButton from "./components/submissions/SubmitButton";
import NextPageContext from "./components/store/nextpageCtx";


function App() {
    const [dataList, setDataList] = useState({ data: [] });
    const [openModal, setOpenModal] = useState(false);
    const [openedVidID, setOpenedVidID] = useState("");
    const nextpageCtx = useContext(NextPageContext);

    const sendKNNreq = (video_id, frame_id, page) => {
        const fetchKNN = async () => {
            const response = await fetch(
                `http://localhost:5000/api?photoId=${video_id}-${frame_id}&page=${1}`
            );
            if (response.ok) {
                const data = await response.json();
                setDataList(data);
                nextpageCtx.setPage(1);
                nextpageCtx.setPhotoId(`${video_id}-${frame_id}`);
                nextpageCtx.setQuery("")
            }
        };
        fetchKNN();
    };

    return (
        <div className={classes.container}>
            <div className={classes.search_space}>
                <Logo />
                <TextQueryForm setDataList={setDataList} />
            </div>
            <div className={classes.result_space}>
                <ImageList
                    dataList={dataList}
                    setDataList={setDataList}
                    sendKNNreq={sendKNNreq}
                    setOpen={setOpenModal}
                    setOpenedVidID={setOpenedVidID}
                />
            </div>
            <VideoModal
                open={openModal}
                setOpen={setOpenModal}
                video_id={openedVidID}
                setVidID={setOpenedVidID}
                sendKNNreq={sendKNNreq}
            />
            <div className={classes.submission_space}>
                <SubmissionList />
                <SubmitButton />
            </div>
        </div>
    );
}

export default App;
