import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import React, { useContext, useState } from "react";
import { Draggable } from "react-beautiful-dnd";
import classes from "./Column.module.css";
import { faTrash } from "@fortawesome/free-solid-svg-icons";
import SubmissionContext from "../store/submissionContext";

const Frame = (props) => {
    const [isHovering, setIsHovering] = useState(false);
    const submissionCtx = useContext(SubmissionContext);

    const handleRemove = () => {
        submissionCtx.removeFrame(props.frame.id);
    };
    return (
        <Draggable draggableId={props.frame.id} index={props.index}>
            {(provided) => (
                <div
                    {...provided.draggableProps}
                    {...provided.dragHandleProps}
                    ref={provided.innerRef}
                    className={classes.frameContainer}
                    onMouseEnter={() => setIsHovering(true)}
                    onMouseLeave={() => setIsHovering(false)}
                >
                    <img
                        className={classes.image}
                        src={`http://localhost:5000/data?video=${props.frame.videoId}&frame=${props.frame.frameId}.jpg`}
                    />
                    {isHovering && (
                        <button className={classes.btn} onClick={handleRemove}>
                            <FontAwesomeIcon icon={faTrash} />
                        </button>
                    )}
                </div>
            )}
        </Draggable>
    );
};

export default Frame;
