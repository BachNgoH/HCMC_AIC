import React, { useEffect, useState } from "react";
import classes from "./FrameDetails.module.css";
import Image from "./Image";

const FrameDetails = ({ video_id, setVidID, sendKNNreq, onClose }) => {
    const [video, setVideo] = useState();
    const [frame, setFrame] = useState();
    useEffect(() => {
        setVideo(video_id.slice(0, 9));
        setFrame(video_id.slice(10));
    }, [video_id]);

    const [allFrames, setAllFrames] = useState([]);
    let data = {};
    try{
        data =  require(`../../../../Metadata_C00_V00/${video_id.slice(0,9)}.json`)
    } catch {
        data['watch_url'] = "No youtube link"
    }

    useEffect(() => {
        const getFrames = async () => {
            const response = await fetch(
                `http://localhost:5000/video/${video_id}.json`
            );
            if (response.ok) {
                const data = await response.json();
                const data_list = data["data"];
                data_list.sort();
                setAllFrames(data_list);
            }
        };
        getFrames();
    }, []);

    return (
        <div className={classes.container}>
            {video && (
                <>
                    <div className={classes.imageSection}>
                        <img
                            src={`http://localhost:5000/data?video=${video}&frame=${frame}.jpg`}
                        />
                        {data && <a href={data['watch_url']} target="_blank">Youtube Link</a>}
                    </div>
                    <div className={classes.frameList}>
                        {allFrames.map((item) => (
                            <Image
                                video_id={item.slice(0, 9)}
                                frame_id={item.slice(10)}
                                setVidID={setVidID}
                                setOpen={() => {}}
                                sendKNNreq={sendKNNreq}
                                setClose={onClose}
                                isChosen={
                                    item === `${video}-${frame}` ? true : false
                                }
                            />
                        ))}
                    </div>
                </>
            )}
        </div>
    );
};

export default FrameDetails;
