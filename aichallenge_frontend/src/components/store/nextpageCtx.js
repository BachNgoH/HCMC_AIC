import React, { useState } from "react";

const NextPageContext = React.createContext({
    page: null,
    query: "",
    photoId: null,
    setPage:() => {},
    setQuery:() => {},
    setPhotoId: () => {}
})

export default NextPageContext;
export const NextPageContextProvider =  ({children}) => {
    const [page, setPage] = useState(null);
    const [query, setQuery] = useState("");
    const [photoId, setPhotoId] = useState("");

    const contextData = {
        page: page,
        setPage: setPage,
        query: query,
        setQuery: setQuery,
        photoId: photoId,
        setPhotoId: setPhotoId
    }

    return <NextPageContext.Provider value={contextData}>
        {children}
    </NextPageContext.Provider>
}