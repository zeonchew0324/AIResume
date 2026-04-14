CREATE TABLE users (                                                                                                                                             
    id UUID PRIMARY KEY,                                                                                                                                         
    email VARCHAR(255) NOT NULL UNIQUE,                                                                                                                          
    created_at TIMESTAMPTZ DEFAULT NOW()                                                                                                                         
);                                                                                                                                                               
                                                                                                                                                                   
CREATE TABLE resumes (                                                                                                                                           
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,                                                                                                
    name VARCHAR(255) NOT NULL,                                                                                                                                  
    resume_text TEXT NOT NULL,                                                                                                                                   
    created_at TIMESTAMPTZ DEFAULT NOW()                                                                                                                         
); 