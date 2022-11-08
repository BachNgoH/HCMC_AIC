import React, { useState } from "react";
import { Modal, Button } from "rsuite";
import FrameDetails from "./FrameDetails";

const VideoModal = ({open, setOpen, video_id, setVidID, sendKNNreq}) => {

    const handleClose = () => setOpen(false)


    return (
        <Modal open={open} onClose={handleClose} backdrop="true" size="full" >
            <Modal.Body>
                
                <FrameDetails video_id={video_id} setVidID={setVidID} sendKNNreq={sendKNNreq} onClose={handleClose}/>
                
            </Modal.Body>
            <Modal.Footer>
                <Button onClick={handleClose} appearance="primary">
                    Ok
                </Button>

            </Modal.Footer>
        </Modal>
    );
};

export default VideoModal;
