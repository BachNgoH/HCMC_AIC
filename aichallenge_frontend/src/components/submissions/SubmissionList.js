import React, { useContext } from "react";
import { DragDropContext } from "react-beautiful-dnd";
import SubmissionContext from "../store/submissionContext";
import Column from "./Column";

const SubmissionList = () => {
    const submissionCtx = useContext(SubmissionContext);

    console.log(submissionCtx.submittedFrames, "IN SUBMISSION LIST");
    const onDragEnd = (result) => {
        const { destination, source, draggableId } = result;

        if (!destination) {
            return;
        }
        if (
            destination.droppableId === source.droppableId &&
            destination.index === source.index
        ) {
            return;
        }

        const column = submissionCtx.columns[source.droppableId];
        const newFrameIds = Array.from(column.frameIds);
        newFrameIds.splice(source.index, 1);
        newFrameIds.splice(destination.index, 0, draggableId);
        console.log(newFrameIds);
        submissionCtx.updateFramesId(newFrameIds)
    };

    return (
        <DragDropContext onDragEnd={onDragEnd}>
            {submissionCtx.columnsOrder.map((columnId) => {
                const column = submissionCtx.columns[columnId];
                const frames = column.frameIds.map(
                    (frameId) => submissionCtx.submittedFrames[frameId]
                );
                console.log(frames);
                return (
                    <Column
                        key={column.id}
                        column={column}
                        frames={frames}
                    />
                );
            })}
        </DragDropContext>
    );
};

export default SubmissionList;
