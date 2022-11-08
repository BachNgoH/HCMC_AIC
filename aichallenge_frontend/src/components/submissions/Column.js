import React from "react";
import { Droppable } from "react-beautiful-dnd";
import Frame from "./Frame";
import classes from "./Column.module.css";

const Column = (props) => {
    return (
        <div className={classes.container}>
            <h3>{props.column.title}</h3>
            <Droppable droppableId={props.column.id}>
                {(provided) => (
                    <div ref={provided.innerRef} {...provided.droppableProps}>
                        {props.frames.map((frame, index) => {
                            console.log(frame);
                            return (
                                frame && (
                                    <Frame
                                        key={frame.id}
                                        frame={frame}
                                        index={index}
                                    />
                                )
                            );
                        })}
                        {provided.placeholder}
                    </div>
                )}
            </Droppable>
        </div>
    );
};

export default Column;
