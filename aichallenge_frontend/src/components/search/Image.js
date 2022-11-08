import React, { useContext, useEffect, useRef, useState } from "react";
import classes from "./Image.module.css";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCheck } from "@fortawesome/free-solid-svg-icons";
import SubmissionContext from "../store/submissionContext";
import NextPageContext from "../store/nextpageCtx";

const Image = ({
    video_id,
    frame_id,
    sendKNNreq,
    setOpen,
    setVidID,
    setClose,
    isChosen,
}) => {
    const [isHovering, setIsHovering] = useState(false);
    const submissionCtx = useContext(SubmissionContext);
    const nextpageCtx = useContext(NextPageContext)

    const imageRef = useRef();
    useEffect(() => {
        if (isChosen) {
            const wait = async () => {
                const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
                await sleep(2000);
                imageRef.current.scrollIntoView({ behavior: "smooth" });
            };
            wait();
        }
    }, [isChosen]);

    const handleSubmission = () => {
        submissionCtx.submitFrame(video_id, frame_id)
    };

    return (
        <div
            className={classes.container}
            onMouseOver={() => setIsHovering(true)}
            onMouseLeave={() => setIsHovering(false)}
            style={isChosen ? { border: "solid 5px green" } : {}}
            ref={imageRef}
        >
            {isHovering && (
                <div className={classes.btns}>
                    {sendKNNreq && (
                        <button
                            className={classes.btn}
                            onClick={() => {
                                sendKNNreq(video_id, frame_id, nextpageCtx.page);
                                setClose();
                            }}
                        >
                            KNN
                        </button>
                    )}
                    <button className={classes.btn} onClick={handleSubmission}>
                        <FontAwesomeIcon icon={faCheck} />
                    </button>
                </div>
            )}
            <img
                className={classes.image}
                src={`KeyFramesC00_V00/${video_id}/${frame_id}.jpg`}
                alt="frame"
            />

            {isHovering && (
                <button
                    className={classes.details}
                    onClick={() => {
                        setOpen(true);
                        setVidID(`${video_id}-${frame_id}`);
                    }}
                    onMouseOver={() => setIsHovering(true)}
                >
                    Details
                </button>
            )}
        </div>
    );
};

export default Image;
