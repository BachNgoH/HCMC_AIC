import React, { useEffect, useState } from "react";

const SubmissionContext = React.createContext({
    submittedFrames: {},
    columns: {},
    columnsOrder: [],
    submitFrame: () => {},
    removeFrame: () => {},
    updateFramesId: () => {},
    clearSubmissions: () => {},
});

export default SubmissionContext;
export const SubmissionProvider = ({ children }) => {
    const [submittedFrames, setSubmittedFrames] = useState(() =>
        localStorage.getItem("submittedFrames")
            ? JSON.parse(localStorage.getItem("submittedFrames"))
            : {}
    );
    const [columns, setColumns] = useState(() =>
        localStorage.getItem("columns")
            ? JSON.parse(localStorage.getItem("columns"))
            : {
                  "column-1": {
                      id: "column-1",
                      title: "Submissions",
                      // frameIds: ["frame-1", "frame-2", "frame-3", "frame-4", "frame-5"],
                      frameIds: [],
                  },
              }
    );
    const [columnsOrder, setColumnsOrder] = useState(["column-1"]);

    const submitFrame = (videoId, frameId) => {
        if (`${videoId}-${frameId}` in submittedFrames) {
            alert("Frame Already submitted");
        } else {
            setSubmittedFrames((submittedFrames) => {
                const newSubmittedFrames = {
                    ...submittedFrames,
                    [`${videoId}-${frameId}`]: {
                        id: `${videoId}-${frameId}`,
                        videoId: videoId,
                        frameId: frameId,
                    },
                };
                localStorage.setItem(
                    "submittedFrames",
                    JSON.stringify(newSubmittedFrames)
                );
                return newSubmittedFrames;
            });

            setColumns((columns) => {
                const columnFrameIds = columns["column-1"].frameIds;
                const newColumnFrameIds = [
                    ...columnFrameIds,
                    `${videoId}-${frameId}`,
                ];

                const newColumns = {
                    ...columns,
                    "column-1": {
                        ...columns["column-1"],
                        frameIds: newColumnFrameIds,
                    },
                };
                localStorage.setItem("columns", JSON.stringify(newColumns));
                return newColumns;
            });
        }
    };

    const deleteFrame = (frameId) => {
        setSubmittedFrames((submittedFrames) => {
            const newSubmittedFrames = submittedFrames;
            delete newSubmittedFrames[frameId];
            localStorage.setItem(
                "submittedFrames",
                JSON.stringify(newSubmittedFrames)
            );
            return newSubmittedFrames;
        });

        setColumns((columns) => {
            const columnFrameIds = columns["column-1"].frameIds;
            const index = columnFrameIds.indexOf(frameId);

            console.log(index, "INDEX");
            if (index > -1) {
                columnFrameIds.splice(index, 1);
            }

            console.log(columnFrameIds);

            const newColumns = {
                ...columns,
                ["column-1"]: {
                    ...columns["column-1"],
                    frameIds: columnFrameIds,
                },
            };
            localStorage.setItem("columns", JSON.stringify(newColumns));
            return newColumns;
        });
    };

    const clearSubmissions = () => {
        const frameIds = columns["column-1"].frameIds;
        frameIds.forEach((frameId) => {
            deleteFrame(frameId);
        });
    };

    const updateSubmittedFramesId = (newFrameIds) => {
        setColumns((columns) => {
            console.log(newFrameIds);
            const newColumn = {
                ...columns["column-1"],
                frameIds: newFrameIds,
            };
            const newColumns = {
                ...columns,
                "column-1": newColumn,
            };
            console.log(newColumns);
            localStorage.setItem("columns", JSON.stringify(newColumns));
            return newColumns;
        });
    };

    const contextData = {
        submittedFrames: submittedFrames,
        columns: columns,
        columnsOrder: columnsOrder,
        submitFrame: submitFrame,
        removeFrame: deleteFrame,
        updateFramesId: updateSubmittedFramesId,
        clearSubmissions: clearSubmissions,
    };

    return (
        <SubmissionContext.Provider value={contextData}>
            {children}
        </SubmissionContext.Provider>
    );
};
