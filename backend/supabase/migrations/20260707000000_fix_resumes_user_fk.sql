-- resumes.user_id comes from the Supabase auth JWT `sub` claim, so it must
-- reference auth.users, not the unused public.users table. Nothing ever
-- inserted into public.users, so the old FK rejected every resume insert.

ALTER TABLE resumes DROP CONSTRAINT IF EXISTS resumes_user_id_fkey;

DROP TABLE IF EXISTS users;

ALTER TABLE resumes
    ADD CONSTRAINT resumes_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES auth.users (id) ON DELETE CASCADE;
