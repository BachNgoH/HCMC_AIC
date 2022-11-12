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
    const nextpageCtx = useContext(NextPageContext);

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
        submissionCtx.submitFrame(video_id, frame_id);
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
                                sendKNNreq(
                                    video_id,
                                    frame_id,
                                    nextpageCtx.page
                                );
                                setClose();
                            }}
                        >
                            KNN
                        </button>
                    )}

                    <button
                        className={classes.btn}
                        onClick={() => {
                            setOpen(true);
                            setVidID(`${video_id}-${frame_id}`);
                        }}
                        onMouseOver={() => setIsHovering(true)}
                    >
                        Details
                    </button>
                    <button className={classes.btn} onClick={handleSubmission}>
                        <FontAwesomeIcon icon={faCheck} />
                    </button>
                </div>
            )}
            <div className={classes.imageContainer}>
            <img
                className={classes.image}
                src={`http://localhost:5000/data?video=${video_id}&frame=${frame_id}.jpg`}
                alt="frame"
                />
            </div>
            {isHovering && <div className={classes.details}>
                {`${video_id}\\${frame_id}`}
            </div>}
        </div>
    );
};

export default Image;
